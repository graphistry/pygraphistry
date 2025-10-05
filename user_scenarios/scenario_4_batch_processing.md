# User Scenario 4: Remote Batch Processing + Persistence

## Current User Pain Point

**Goal**: Process multiple large datasets remotely in batch jobs and persist results for later analysis

**Current Workflow (Inefficient for Batch)**:
```python
import graphistry
import pandas as pd
from datetime import datetime, timedelta

# Batch processing scenario - daily fraud detection analysis
date_ranges = [
    '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05'
]

processed_datasets = []

for date in date_ranges:
    # Step 1: Load daily transaction data
    daily_transactions = load_transaction_data(date)
    g = graphistry.edges(daily_transactions, 'sender', 'receiver')

    # Step 2: Remote fraud detection analysis
    fraud_analysis = g.gfql_remote([
        call('materialize_nodes'),
        call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'centrality'}),
        call('compute_cugraph', {'alg': 'louvain', 'out_col': 'cluster'}),
        call('encode_point_color', {'column': 'cluster', 'as_categorical': True}),
        call('encode_point_size', {'column': 'centrality'}),
        call('name', {'name': f'Fraud Analysis {date}'}),
        call('description', {'description': f'Daily fraud detection analysis for {date}'})
    ])

    # Step 3: Manual persistence (inefficient!)
    uploaded = fraud_analysis.plot()  # ❌ Unnecessary upload for batch job
    processed_datasets.append({
        'date': date,
        'dataset_id': uploaded._dataset_id,
        'url': uploaded._url
    })

# Step 4: Create batch summary
print(f"Processed {len(processed_datasets)} datasets")
for dataset in processed_datasets:
    print(f"{dataset['date']}: {dataset['dataset_id']}")
```

**Problems**:
1. **Unnecessary uploads**: Batch jobs don't need immediate visualization
2. **Resource waste**: plot() is overkill for automated workflows
3. **Slow feedback**: Serial uploads delay batch completion
4. **Complex tracking**: Manual dataset_id collection

## Desired Workflow (Efficient Batch Processing)

**Option A: Batch-optimized with call('save')**
```python
# Efficient batch processing with automatic persistence
processed_datasets = []

for date in date_ranges:
    daily_transactions = load_transaction_data(date)
    g = graphistry.edges(daily_transactions, 'sender', 'receiver')

    # Single operation: analysis + persistence
    fraud_analysis = g.gfql_remote([
        call('materialize_nodes'),
        call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'centrality'}),
        call('compute_cugraph', {'alg': 'louvain', 'out_col': 'cluster'}),
        call('encode_point_color', {'column': 'cluster', 'as_categorical': True}),
        call('encode_point_size', {'column': 'centrality'}),
        call('name', {'name': f'Fraud Analysis {date}'}),
        call('description', {'description': f'Daily fraud detection analysis for {date}'}),
        call('save')  # 🎯 Efficient server-side persistence
    ])

    processed_datasets.append({
        'date': date,
        'dataset_id': fraud_analysis._dataset_id,  # ✅ Available immediately
        'url': fraud_analysis.url(),              # ✅ Generate URL without upload
        'summary': summarize_fraud_indicators(fraud_analysis)
    })

# Batch job complete - all datasets persisted and trackable
print(f"Batch job complete: {len(processed_datasets)} datasets processed")
```

**Option B: Parallel batch with persist=True**
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

def process_daily_fraud_analysis(date):
    """Process single day's fraud analysis with persistence"""
    daily_transactions = load_transaction_data(date)
    g = graphistry.edges(daily_transactions, 'sender', 'receiver')

    return g.gfql_remote([
        call('materialize_nodes'),
        call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'centrality'}),
        call('compute_cugraph', {'alg': 'louvain', 'out_col': 'cluster'}),
        call('encode_point_color', {'column': 'cluster'}),
        call('name', {'name': f'Fraud Analysis {date}'}),
    ], persist=True)  # 🎯 Enable persistence for batch workflow

# Parallel processing for faster batch completion
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_daily_fraud_analysis, date_ranges))

# All results have dataset_id for immediate use
dataset_ids = [result._dataset_id for result in results]
print(f"Parallel batch complete: {dataset_ids}")
```

## Advanced Batch Processing Patterns

**A. Multi-stage pipeline with intermediate persistence**:
```python
class FraudDetectionPipeline:
    def __init__(self):
        self.stage_results = {}

    def stage_1_preprocessing(self, date):
        """Stage 1: Data preprocessing and basic graph construction"""
        raw_data = load_transaction_data(date)
        g = graphistry.edges(raw_data, 'sender', 'receiver')

        result = g.gfql_remote([
            call('materialize_nodes'),
            call('filter_edges_by_dict', {'filter_dict': {'amount': {'$gt': 1000}}}),  # High-value only
            call('name', {'name': f'Preprocessed Transactions {date}'}),
            call('save')  # 🎯 Persist intermediate results
        ])

        self.stage_results[f'{date}_preprocessed'] = result._dataset_id
        return result

    def stage_2_analysis(self, preprocessed_result, date):
        """Stage 2: Advanced fraud detection algorithms"""
        # Continue from preprocessed dataset
        result = preprocessed_result.gfql_remote([
            call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'risk_score'}),
            call('compute_cugraph', {'alg': 'louvain', 'out_col': 'fraud_cluster'}),
            call('encode_point_color', {'column': 'fraud_cluster'}),
            call('encode_point_size', {'column': 'risk_score'}),
            call('name', {'name': f'Fraud Analysis Final {date}'}),
            call('save')  # 🎯 Persist final results
        ])

        self.stage_results[f'{date}_final'] = result._dataset_id
        return result

    def run_pipeline(self, date_ranges):
        """Run complete pipeline for multiple dates"""
        final_results = []

        for date in date_ranges:
            # Stage 1
            preprocessed = self.stage_1_preprocessing(date)

            # Stage 2
            final = self.stage_2_analysis(preprocessed, date)
            final_results.append(final)

        return final_results

# Run pipeline
pipeline = FraudDetectionPipeline()
results = pipeline.run_pipeline(date_ranges)

# Access intermediate and final results
print("Pipeline stage tracking:")
for stage, dataset_id in pipeline.stage_results.items():
    print(f"  {stage}: {dataset_id}")
```

**B. Automated monitoring and alerting**:
```python
class AutomatedFraudMonitoring:
    def __init__(self, alert_threshold=0.8):
        self.alert_threshold = alert_threshold
        self.baseline_dataset_id = None

    def daily_fraud_check(self, date):
        """Automated daily fraud detection with alerting"""
        daily_data = load_transaction_data(date)
        g = graphistry.edges(daily_data, 'sender', 'receiver')

        # Analysis with automatic persistence
        result = g.gfql_remote([
            call('materialize_nodes'),
            call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'risk_score'}),
            call('name', {'name': f'Daily Fraud Monitor {date}'}),
            call('save')
        ])

        # Check for anomalies
        high_risk_nodes = result._nodes[result._nodes['risk_score'] > self.alert_threshold]

        if len(high_risk_nodes) > 0:
            # Generate alert with persistent visualization
            alert_message = f"High-risk activity detected on {date}"
            alert_url = result.url()

            send_fraud_alert(alert_message, alert_url, high_risk_nodes.to_dict('records'))

        # Update baseline for trend analysis
        if self.baseline_dataset_id is None:
            self.baseline_dataset_id = result._dataset_id

        return result

    def weekly_trend_analysis(self, week_results):
        """Analyze weekly trends from daily results"""
        dataset_ids = [r._dataset_id for r in week_results]

        # Create trend comparison dashboard
        trend_dashboard_url = create_trend_dashboard(dataset_ids)

        return {
            'trend_url': trend_dashboard_url,
            'dataset_ids': dataset_ids,
            'summary': 'Weekly fraud trend analysis complete'
        }

# Automated monitoring setup
monitor = AutomatedFraudMonitoring()

# Daily monitoring (runs automatically)
daily_results = []
for date in date_ranges:
    result = monitor.daily_fraud_check(date)
    daily_results.append(result)

# Weekly summary
weekly_summary = monitor.weekly_trend_analysis(daily_results)
print(f"Weekly analysis: {weekly_summary['trend_url']}")
```

## Benefits for Production Batch Processing

1. **Efficient resource usage**: No unnecessary uploads in automated workflows
2. **Parallel processing**: persist=True works well with concurrent execution
3. **Audit trails**: All intermediate and final results are persistently stored
4. **API integration**: dataset_id enables downstream system integration
5. **Monitoring ready**: Persistent URLs for alerts and notifications
6. **Cost optimization**: Single server operations reduce compute and transfer costs

## Production Deployment Patterns

```python
# Kubernetes job configuration
def create_batch_fraud_job():
    return {
        'apiVersion': 'batch/v1',
        'kind': 'Job',
        'spec': {
            'template': {
                'spec': {
                    'containers': [{
                        'name': 'fraud-analysis',
                        'image': 'fraud-detector:latest',
                        'env': [
                            {'name': 'GRAPHISTRY_PERSIST', 'value': 'true'},
                            {'name': 'BATCH_SIZE', 'value': '5'}
                        ]
                    }]
                }
            }
        }
    }

# Environment-aware configuration
def get_processing_config():
    if os.environ.get('PRODUCTION'):
        return {'persist': True, 'parallel_workers': 10}
    else:
        return {'persist': False, 'parallel_workers': 2}  # Development
```