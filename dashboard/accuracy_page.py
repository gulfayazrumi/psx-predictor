"""
Model Accuracy Page - Track prediction performance
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

def render():
    st.title("🎯 Model Accuracy Tracker")
    st.markdown("Track the performance of AI predictions against actual market outcomes")
    
    # Load accuracy report
    report_path = Path("reports/accuracy_report.json")
    
    if not report_path.exists():
        st.warning("⚠️ No accuracy data available yet. Predictions need at least one day to verify against actual outcomes.")
        st.info("💡 The system will automatically track accuracy as predictions are made daily and verified against next-day market data.")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    summary = report.get('summary', {})
    detailed = report.get('detailed_results', [])
    by_confidence = report.get('by_confidence', {})
    
    # Summary Metrics
    st.markdown("### 📊 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predictions",
            summary.get('total_predictions', 0),
            help="Number of predictions that have been verified"
        )
    
    with col2:
        accuracy = summary.get('accuracy_pct', 0)
        st.metric(
            "Direction Accuracy",
            f"{accuracy:.1f}%",
            help="Percentage of correct UP/DOWN predictions"
        )
    
    with col3:
        st.metric(
            "Correct Predictions",
            summary.get('correct_predictions', 0),
            help="Number of predictions with correct direction"
        )
    
    with col4:
        avg_error = summary.get('avg_price_error_pct', 0)
        st.metric(
            "Avg Price Error",
            f"{avg_error:.2f}%",
            help="Average percentage difference between predicted and actual price"
        )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Detailed Log", "📈 Confidence Analysis", "📅 Performance Timeline"])
    
    with tab1:
        st.markdown("### Prediction Verification Log")
        
        if detailed:
            df_detailed = pd.DataFrame(detailed)
            
            # Add Pass/Fail column
            df_detailed['Result'] = df_detailed['direction_correct'].apply(
                lambda x: '✅ PASS' if x else '❌ FAIL'
            )
            
            # Format columns for display
            display_df = df_detailed[[
                'symbol', 'prediction_date', 'target_date', 'Result',
                'predicted_direction', 'actual_direction',
                'current_price_at_pred', 'predicted_price', 'actual_price',
                'price_error_pct', 'confidence', 'confidence_bucket'
            ]].copy()
            
            display_df.columns = [
                'Symbol', 'Prediction Date', 'Target Date', 'Result',
                'Predicted', 'Actual', 'Price at Prediction',
                'Predicted Price', 'Actual Price', 'Price Error %',
                'Confidence', 'Confidence Level'
            ]
            
            # Round numeric columns
            display_df['Price at Prediction'] = display_df['Price at Prediction'].round(2)
            display_df['Predicted Price'] = display_df['Predicted Price'].round(2)
            display_df['Actual Price'] = display_df['Actual Price'].round(2)
            display_df['Price Error %'] = display_df['Price Error %'].round(2)
            display_df['Confidence'] = (display_df['Confidence'] * 100).round(1)
            
            # Color code results
            def highlight_result(row):
                if row['Result'] == '✅ PASS':
                    return ['background-color: #d4edda'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_result, axis=1),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Log",
                data=csv,
                file_name=f"accuracy_log_{report.get('generated_at', 'latest').replace(' ', '_').replace(':', '-')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No detailed predictions available yet.")
    
    with tab2:
        st.markdown("### Accuracy by Confidence Level")
        
        if by_confidence:
            # Create DataFrame for confidence analysis
            conf_data = []
            for bucket, stats in by_confidence.items():
                conf_data.append({
                    'Confidence Level': bucket,
                    'Total': stats['total'],
                    'Correct': stats['correct'],
                    'Accuracy %': stats['accuracy'],
                    'Avg Error %': stats['avg_error']
                })
            
            df_conf = pd.DataFrame(conf_data)
            
            # Bar chart for accuracy by confidence
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=df_conf['Confidence Level'],
                y=df_conf['Accuracy %'],
                name='Accuracy',
                marker_color='#28a745',
                text=df_conf['Accuracy %'].round(1),
                textposition='outside',
                texttemplate='%{text}%'
            ))
            
            fig.update_layout(
                title="Direction Accuracy by Confidence Level",
                xaxis_title="Confidence Level",
                yaxis_title="Accuracy (%)",
                yaxis_range=[0, 100],
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Table view
            st.markdown("#### Detailed Breakdown")
            st.dataframe(df_conf, use_container_width=True)
            
            # Insights
            st.markdown("#### 💡 Insights")
            if df_conf['Total'].sum() > 0:
                best_bucket = df_conf.loc[df_conf['Accuracy %'].idxmax(), 'Confidence Level']
                best_accuracy = df_conf['Accuracy %'].max()
                st.success(f"**Best Performance:** {best_bucket} confidence predictions with {best_accuracy:.1f}% accuracy")
                
                if 'High (70%+)' in by_confidence:
                    high_conf_acc = by_confidence['High (70%+)']['accuracy']
                    st.info(f"**High Confidence Predictions:** {high_conf_acc:.1f}% accuracy - These are the most reliable signals")
        else:
            st.info("Not enough data for confidence analysis yet.")
    
    with tab3:
        st.markdown("### Performance Over Time")
        
        if detailed and len(detailed) > 1:
            df_timeline = pd.DataFrame(detailed)
            df_timeline['prediction_date'] = pd.to_datetime(df_timeline['prediction_date'])
            df_timeline = df_timeline.sort_values('prediction_date')
            
            # Calculate cumulative accuracy
            df_timeline['cumulative_correct'] = df_timeline['direction_correct'].cumsum()
            df_timeline['cumulative_total'] = range(1, len(df_timeline) + 1)
            df_timeline['cumulative_accuracy'] = (df_timeline['cumulative_correct'] / df_timeline['cumulative_total']) * 100
            
            # Line chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_timeline['prediction_date'],
                y=df_timeline['cumulative_accuracy'],
                mode='lines+markers',
                name='Cumulative Accuracy',
                line=dict(color='#007bff', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Accuracy Trend Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Accuracy (%)",
                yaxis_range=[0, 100],
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Timeline analysis will be available after more predictions are verified (need at least 2 data points)")
    
    # Footer
    st.markdown("---")
    st.caption(f"📅 Last Updated: {report.get('generated_at', 'N/A')}")
    st.caption("💡 Tip: Run `python src/analysis/accuracy_tracker.py` to manually update accuracy metrics")

if __name__ == "__main__":
    render()
