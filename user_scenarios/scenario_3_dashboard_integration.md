# User Scenario 3: Remote Algorithm + Dashboard Integration

## Current User Pain Point

**Goal**: Run graph algorithms remotely and integrate results into live business dashboard

**Current Workflow (Dashboard Integration Friction)**:
```python
import graphistry
import pandas as pd

# Business network data - customer relationships, influence analysis
network_df = pd.read_csv('customer_network.csv')
g = graphistry.edges(network_df, 'customer_a', 'customer_b')

# Step 1: Remote algorithm execution
influence_result = g.gfql_remote([
    call('materialize_nodes'),
    call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'influence_score'}),
    call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'}),
    call('encode_point_color', {'column': 'community', 'as_categorical': True}),
    call('encode_point_size', {'column': 'influence_score'}),
    call('name', {'name': 'Customer Influence Network'}),
    call('description', {'description': 'PageRank influence scores with community detection'})
])

# Step 2: Dashboard wants the dataset_id, but we need to upload first
uploaded = influence_result.plot()  # ❌ Forces another upload
dataset_id = uploaded._dataset_id   # Finally get the ID we need

# Step 3: Dashboard integration
dashboard_api_call(dataset_id, 'customer_influence_widget')
```

**Problems**:
1. **Unnecessary upload**: Dashboard just needs dataset_id, not visualization
2. **API complexity**: Extra step for integration
3. **Real-time friction**: Delays in live dashboard updates

## Desired Workflow (Direct Integration)

**Option A: Algorithm + Save for direct API integration**
```python
# Single operation optimized for API integration
influence_result = g.gfql_remote([
    call('materialize_nodes'),
    call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'influence_score'}),
    call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'}),
    call('encode_point_color', {'column': 'community', 'as_categorical': True}),
    call('encode_point_size', {'column': 'influence_score'}),
    call('name', {'name': 'Customer Influence Network'}),
    call('description', {'description': 'PageRank influence scores with community detection'}),
    call('save')  # 🎯 Persist for immediate API use
])

# Direct dashboard integration - no extra upload needed
dataset_id = influence_result._dataset_id
dashboard_api_call(dataset_id, 'customer_influence_widget')

# Bonus: Data still available for local analysis
top_influencers = influence_result._nodes.nlargest(10, 'influence_score')
print(f"Top influencers: {top_influencers['customer_id'].tolist()}")
```

**Option B: Automated refresh workflows**
```python
def refresh_customer_influence_dashboard():
    """Automated function for daily dashboard refresh"""
    network_data = get_latest_customer_network()
    g = graphistry.edges(network_data, 'customer_a', 'customer_b')

    # Refresh with consistent naming for dashboard tracking
    result = g.gfql_remote([
        call('materialize_nodes'),
        call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'influence_score'}),
        call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'}),
        call('encode_point_color', {'column': 'community'}),
        call('encode_point_size', {'column': 'influence_score'}),
        call('name', {'name': f'Customer Influence {date.today()}'}),
    ], persist=True)  # 🎯 Automated persistence

    # Update dashboard configuration
    update_dashboard_widget('customer_influence', result._dataset_id)

    return result._dataset_id

# Scheduled execution
daily_dataset_id = refresh_customer_influence_dashboard()
print(f"Dashboard updated with dataset: {daily_dataset_id}")
```

## Advanced Dashboard Integration Patterns

**A. Multi-algorithm comparison dashboard**:
```python
algorithms = ['pagerank', 'betweenness_centrality', 'closeness_centrality']
algorithm_datasets = {}

for alg in algorithms:
    result = g.gfql_remote([
        call('materialize_nodes'),
        call('compute_cugraph', {'alg': alg, 'out_col': f'{alg}_score'}),
        call('encode_point_size', {'column': f'{alg}_score'}),
        call('name', {'name': f'Customer Network - {alg.title()}'}),
        call('save')
    ])
    algorithm_datasets[alg] = result._dataset_id

# Dashboard gets all datasets for comparison view
dashboard_comparison_widget(algorithm_datasets)
```

**B. Real-time monitoring with thresholds**:
```python
def monitor_network_changes(threshold=0.1):
    """Monitor for significant network changes"""
    current_result = g.gfql_remote([
        call('compute_cugraph', {'alg': 'pagerank'}),
        call('name', {'name': f'Network Monitor {datetime.now()}'}),
        call('save')
    ])

    # Compare with baseline if exists
    if baseline_dataset_id := get_baseline_dataset():
        comparison_url = create_comparison_dashboard(baseline_dataset_id, current_result._dataset_id)

        # Alert if significant changes detected
        changes = detect_network_changes(baseline_dataset_id, current_result._dataset_id)
        if changes > threshold:
            send_alert(f"Network changes detected: {changes:.2%}", comparison_url)

    return current_result._dataset_id
```

## Business Intelligence Integration

**API-first design for BI tools**:
```python
class CustomerNetworkAnalytics:
    def __init__(self, graphistry_client):
        self.g = graphistry_client

    def generate_influence_report(self, date_range):
        """Generate influence analysis for BI dashboard"""
        network_data = self.get_network_data(date_range)
        g = self.g.edges(network_data, 'customer_a', 'customer_b')

        result = g.gfql_remote([
            call('materialize_nodes'),
            call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'influence'}),
            call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'}),
            call('encode_point_color', {'column': 'community'}),
            call('name', {'name': f'Influence Report {date_range}'}),
            call('save')
        ])

        # Return structured data for BI tools
        return {
            'dataset_id': result._dataset_id,
            'visualization_url': result.url(),
            'summary_stats': self.calculate_summary_stats(result),
            'top_influencers': result._nodes.nlargest(10, 'influence').to_dict('records')
        }

    def calculate_summary_stats(self, result):
        nodes = result._nodes
        return {
            'total_customers': len(nodes),
            'communities_detected': nodes['community'].nunique(),
            'avg_influence': nodes['influence'].mean(),
            'influence_distribution': nodes['influence'].describe().to_dict()
        }

# BI tool integration
analytics = CustomerNetworkAnalytics(graphistry)
report = analytics.generate_influence_report('2024-Q3')

# Direct integration with Tableau, PowerBI, etc.
bi_dashboard.update_widget('customer_influence', report['dataset_id'])
slack_notification(f"Q3 report ready: {report['visualization_url']}")
```

## Benefits for Business Intelligence

1. **API-first**: dataset_id enables direct BI tool integration
2. **Automated workflows**: persist=True perfect for scheduled reports
3. **Real-time updates**: Immediate dashboard refresh without manual steps
4. **Hybrid analysis**: Server-side computation + local summary statistics
5. **Audit trail**: Persistent datasets enable historical comparison