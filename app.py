import streamlit as st
import boto3
import pandas as pd
from decimal import Decimal
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Supercloud Inference Monitor", layout="wide")
st.title("🎛️ Datacenter Job Failure Prediction Monitor")

# Add spacing
st.write("<br>", unsafe_allow_html=True)

# 2. Data Fetching
@st.cache_data(ttl=60)
def fetch_dynamodb_data():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('SupercloudJobPredictions') 
    
    response = table.scan()
    items = response.get('Items', [])
    
    if not items:
        return pd.DataFrame()
        
    df = pd.DataFrame(items)
    
    if 'metadata' in df.columns:
        metadata_df = pd.json_normalize(df['metadata'])
        df = pd.concat([df.drop(columns=['metadata']), metadata_df], axis=1)
    
    for col in df.columns:
        df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        
    return df

# 3. Load and Process Data
df = fetch_dynamodb_data()

if df.empty:
    st.warning("No tracking data found in DynamoDB yet. Run a test event in Lambda!")
else:

    # --- Top Level Metrics ---
    total_jobs = len(df)
    failed_jobs = int((df['prediction'] == 0).sum()) 

    # --- Calculate Deployed Model Accuracy (Using raw DynamoDB 'df') ---
    if 'prediction' in df.columns and 'state' in df.columns:

    # Convert state column to numeric to resolve string vs integer mismatch
        # errors='coerce' turns non-numeric values into NaN
        df['state_numeric'] = pd.to_numeric(df['state'], errors='coerce')

        # Ensure we account for both int and float variations from DynamoDB parsing
        failure_states = [5, 6, 7, 11]
        success_states = [3]
    
        # Mask for Correct Predictions
        correct_failures = (df['prediction'] == 0) & (df['state_numeric'].isin(failure_states))
        correct_successes = (df['prediction'] == 1) & (df['state_numeric'].isin(success_states))
        total_correct = (correct_failures | correct_successes).sum()
    
        # Mask for Incorrect Predictions
        incorrect_failures = (df['prediction'] == 0) & (df['state_numeric'].isin(success_states))
        incorrect_successes = (df['prediction'] == 1) & (df['state_numeric'].isin(failure_states))
        total_incorrect = (incorrect_failures | incorrect_successes).sum()
    
        # Total evaluable jobs (this naturally ignores cancelled jobs, state 4)
        total_evaluable = total_correct + total_incorrect
    
        if total_evaluable > 0:
            model_accuracy = (total_correct / total_evaluable) * 100
        else:
            model_accuracy = 100.0 # Default
    else:
        total_correct = 0
        total_incorrect = 0
        total_evaluable = 0
        model_accuracy = 100.0

    # Calculate Total Hours Saved (Duration - 2 hours for predicted failures)
    # Only calculate if the column exists and has data
    if 'duration_hours' in df.columns:
        failure_mask = df['prediction'] == 0
        # If duration - 2 is negative (e.g. job crashed immediately), treat it as 0 hours saved
        hours_saved_series = (df.loc[failure_mask, 'duration_hours'] - 2).clip(lower=0)
        total_hours_saved = hours_saved_series.sum()
    else:
        total_hours_saved = 0.0

    # Expand to 4 columns to fit the new tracker
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs Evaluated", total_jobs)
    col2.metric("Predicted Job Failures", failed_jobs)
    col3.metric("Deployed Model Accuracy", f"{model_accuracy:.1f}%")
    col4.metric("Total Compute Hours Saved", f"{int(round(total_hours_saved))} hrs", delta="⚡ Efficiency Gain")

    st.markdown("---")

    # --- VISUAL 1: Accuracy Drift Over Time ---
    # Convert timestamp to datetime if needed
    df['prediction_timestamp'] = pd.to_datetime(df['prediction_timestamp'])
    df['date'] = df['prediction_timestamp'].dt.date
    df = df.sort_values('prediction_timestamp')

    evaluable_mask = (df['state_numeric'] != 4)

    # Define 'is_correct' for the trend calculation
    df['is_correct'] = (
        ((df['prediction'] == 0) & (df['state_numeric'].isin([5, 6, 7, 11]))) |
        ((df['prediction'] == 1) & (df['state_numeric'] == 3))
    ).astype(int)

    evaluable_df = df[evaluable_mask].copy()

    # Group and calculate 7-day rolling average
    daily_stats = evaluable_df.groupby('date').agg(
        daily_correct=('is_correct', 'sum'),
        daily_total=('is_correct', 'count')
    ).reset_index()

    daily_stats['accuracy_rolling'] = (
        daily_stats['daily_correct'].rolling(window=7, min_periods=1).sum() / 
        daily_stats['daily_total'].rolling(window=7, min_periods=1).sum()
    ) * 100

    daily_stats['accuracy_rolling'] = daily_stats['accuracy_rolling'].round(2)

    # Keep only last 10 days
    recent_stats = daily_stats.tail(10)

    # Create Figures
    # Trend Chart
    fig_drift = px.line(recent_stats, x='date', y='accuracy_rolling', 
                        title="Model Accuracy (7-Day Rolling Average)",
                        labels={'accuracy_rolling': 'Accuracy (%)', 'date': 'Date'},
                        markers=True)
    fig_drift.update_layout(yaxis_range=[0, 100], margin=dict(l=20, r=20, t=50, b=20))


    # --- VISUAL 2: Prediction Confidence Histogram ---
    # We assume 'confidence_percentage' is a numeric column (0-100)
    if 'confidence_percentage' in df.columns:
        fig_conf = px.histogram(df, x='confidence_percentage', 
                                nbins=20,
                                title="Distribution of Model Confidence Scores",
                                labels={'confidence_percentage': 'Confidence Score (%)'},
                                color_discrete_sequence=['#636EFA'])
        fig_conf.update_layout(bargap=0.1)
    #    st.plotly_chart(fig_conf, use_container_width=True)
    else:
        st.info("Confidence data not available in raw DataFrame.")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_drift, use_container_width=True)

    with col2:
        st.plotly_chart(fig_conf, use_container_width=True)


    # --- Raw Data Table Formatting ---
   # st.subheader("📋 Live Prediction Audit Trail")
    
    # Create a copy so we don't mess up the chart data
    display_df = df.sort_values(by='prediction_timestamp', ascending=False).copy()
    display_df = display_df.head(100)
    
    # Convert UTC to Local Eastern Time and make readable
    display_df['prediction_timestamp'] = pd.to_datetime(display_df['prediction_timestamp'])
    
    # Tell pandas it's currently UTC, then convert to Eastern Time
    if display_df['prediction_timestamp'].dt.tz is None:
        display_df['prediction_timestamp'] = display_df['prediction_timestamp'].dt.tz_localize('UTC')
        
    display_df['prediction_timestamp'] = (display_df['prediction_timestamp']
                                          .dt.tz_convert('US/Eastern')
                                          .dt.strftime('%Y-%m-%d %I:%M %p')) # Format as YYYY-MM-DD HH:MM AM/PM
    
    # Map predictions to strings
    display_df['prediction'] = display_df['prediction'].map({0: 'Job Failure', 1: 'Job Completed'})
    
    # Format Confidence Percentage as a whole number
    if 'confidence_percentage' in display_df.columns:
        clean_conf = display_df['confidence_percentage'].astype(str).str.strip()
        clean_conf = clean_conf.str.replace('%', '', regex=False)
        
        # Format as :0.0f to drop the decimal places entirely
        display_df['confidence_percentage'] = pd.to_numeric(clean_conf, errors='coerce').apply(
            lambda x: f"{x:0.0f}%" if pd.notnull(x) else "0%"
        )
    
    # Format Duration to HH:MM
    if 'duration_hours' in display_df.columns:
        display_df['duration'] = display_df['duration_hours'].apply(
            lambda x: f"{int(x):02d}:{int(round((x*60)%60)):02d}" if pd.notnull(x) else "00:00"
        )
        
    # Rename state
    display_df = display_df.rename(columns={'state': 'Actual End State'})

    # Map cluster codes to readable Actual End States
    if 'Actual End State' in display_df.columns:
        status_mapping = {
            3: "Job Completed",
            4: "Job Cancelled",
            5: "Job Failure (Crashed)",
            6: "Job Failure (Timeout)",
            11: "Job Failure (Out of Memory)",
            7: "Node Failed"
        }

        display_df['Actual End State'] = pd.to_numeric(display_df['Actual End State'], errors='coerce')
        display_df['Actual End State'] = display_df['Actual End State'].map(status_mapping).fillna("Unknown State") 

    # --- CALCULATE ACCURACY STATUS ---
    def evaluate_prediction_accuracy(row):
        pred = row['prediction']         
        actual = row['Actual End State']  

        if "Cancelled" in actual:
            return "N/A ➖"
        
        if pred == "Job Completed" and actual == "Job Completed":
            return "Correct ✅"
        elif pred == "Job Failure" and "Failure" in actual:
            return "Correct ✅"
        else:
            return "Incorrect ❌"

    display_df['Prediction Correct'] = display_df.apply(evaluate_prediction_accuracy, axis=1)
    
    # Calculate and format Time Saved per job
    if 'duration_hours' in display_df.columns and 'prediction' in display_df.columns:
        def calculate_row_time_saved(row):
            if row['Prediction Correct'] == "Incorrect ❌":
                return "N/A"
            if row['prediction'] == 'Job Failure' and pd.notnull(row['duration_hours']):
                saved_hrs = row['duration_hours'] - 2
                if saved_hrs <= 0:
                    return "00:00"
                return f"{int(saved_hrs):02d}:{int(round((saved_hrs*60)%60)):02d}"
            else:
                return "N/A"
                
        display_df['Time Saved'] = display_df.apply(calculate_row_time_saved, axis=1)


    # Drop unwanted columns 
    cols_to_drop = ['source_file_path', 'duration_hours', 'Status']
    display_df = display_df.drop(columns=[c for c in cols_to_drop if c in display_df.columns], errors='ignore')
    
    # Enforce your specific column order
    preferred_order = [
        'id_job', 
        'prediction_timestamp', 
        'prediction', 
        'confidence_percentage', 
        'Actual End State', 
        'Prediction Correct', 
        'duration', 
        'Time Saved',
        'allocated_cpus',
        'allocated_gpus'
    ]
    # Filter preferred order to only include columns that actually exist in the dataframe
    existing_preferred = [c for c in preferred_order if c in display_df.columns]
    remaining_cols = [c for c in display_df.columns if c not in existing_preferred] 
    display_df = display_df[existing_preferred + remaining_cols]

    # --- LATEST JOB SECTION ---
    if not display_df.empty:
        # Slice the newest job (index 0) and the history (index 1 to end)
        latest_job = display_df.iloc[0]
        history_df = display_df.iloc[1:]

        # Create a visually appealing bordered container for the latest intercept
        with st.container(border=True):
            st.subheader("⚡ Latest Pipeline Evaluation")
            
            # Use columns to put the most critical metrics front and center
            hero_col1, hero_col2, hero_col3 = st.columns(3)
            
            # Add dynamic emojis based on the prediction outcome
            status_icon = "🛑" if latest_job['prediction'] == 'Job Failure' else "✅"
            
            hero_col1.metric(f"Prediction", f"{status_icon} {latest_job['prediction']}")
            hero_col2.metric("Model Confidence", latest_job['confidence_percentage'])
            hero_col3.metric("Compute Time Saved (hrs:mins)", latest_job['Time Saved'])
            
            # Add a footer for the secondary details
            st.caption(f"**Processed:** {latest_job['prediction_timestamp']} | **Actual End State:** {latest_job['Actual End State']} | **Job Duration:** {latest_job['duration']}")

        st.markdown("<br>", unsafe_allow_html=True) # Add a little breathing room

        # --- HISTORY TABLE ---
        st.subheader("📋 Historical Audit Trail")
        
        # Mapping dict for your specified column names
        rename_mapping = {
            'prediction_timestamp': 'Prediction Timestamp',
            'prediction': 'Prediction',
            'confidence_percentage': 'Model Confidence',
            'duration': 'Actual Duration (hrs:mins)',
            'id_job': 'Job ID',
            'allocated_cpus': "CPU's Used",
            'allocated_gpus': "GPU's Used",
            'Prediction Correct': 'Prediction Accuracy',
            'Time Saved' : 'Time Saved (hrs:mins)'
        }
        
        # Isolate the historical rows, select the ordered columns, and rename them
        history_display_df = history_df[existing_preferred].rename(columns=rename_mapping)
        
        st.dataframe(history_display_df, use_container_width=True, hide_index=True)
