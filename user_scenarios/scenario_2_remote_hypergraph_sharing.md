# User Scenario 2: Remote Hypergraph + Team Sharing Workflow

## Current User Pain Point

**Goal**: Transform transaction data into hypergraph remotely and share with team dashboard

**Current Workflow (Complex)**:
```python
import graphistry
import pandas as pd

# Transaction event data - complex multi-entity relationships
transactions_df = pd.read_csv('transaction_events.csv')  # 100k+ rows
g = graphistry.nodes(transactions_df)

# Step 1: Remote hypergraph transformation
hypergraph_result = g.gfql_remote([
    call('hypergraph', {
        'entity_types': ['user_id', 'merchant_id', 'product_id'],
        'direct': True,
        'engine': 'cudf'
    }),
    call('encode_point_color', {'column': 'type', 'as_categorical': True}),
    call('name', {'name': 'Transaction Network'}),
    call('description', {'description': 'Hypergraph of user-merchant-product interactions'})
])

# Step 2: Manual upload for team sharing
shared_viz = hypergraph_result.plot()  # ❌ Another upload!

# Step 3: Share with team
team_dashboard_url = f"https://company-dashboard.com/embed?viz={shared_viz}"
```

**Problems**:
1. **Redundant processing**: Server transforms, client downloads, server stores again
2. **Team workflow friction**: Extra step breaks automated dashboard integration
3. **Resource inefficiency**: Large hypergraphs transfer twice

## Desired Workflow (Streamlined)

**Option A: Automatic team sharing with call('save')**
```python
# Single operation with immediate sharing capability
hypergraph_result = g.gfql_remote([
    call('hypergraph', {
        'entity_types': ['user_id', 'merchant_id', 'product_id'],
        'direct': True,
        'engine': 'cudf'
    }),
    call('encode_point_color', {'column': 'type', 'as_categorical': True}),
    call('name', {'name': 'Transaction Network'}),
    call('description', {'description': 'Hypergraph of user-merchant-product interactions'}),
    call('save')  # 🎯 Server-side persistence
])

# Immediate integration with team systems
dataset_id = hypergraph_result._dataset_id
viz_url = hypergraph_result.url()

# Direct dashboard integration
team_dashboard_url = f"https://company-dashboard.com/embed?dataset={dataset_id}"
print(f"Dashboard ready: {team_dashboard_url}")
```

**Option B: Batch processing with persist=True**
```python
# For automated daily/weekly reports
def create_transaction_hypergraph(date_range):
    transactions = get_transaction_data(date_range)
    g = graphistry.nodes(transactions)

    return g.gfql_remote([
        call('hypergraph', {'entity_types': ['user_id', 'merchant_id', 'product_id']}),
        call('encode_point_color', {'column': 'type'}),
        call('name', {'name': f'Transactions {date_range}'}),
    ], persist=True)  # 🎯 Enable persistence for automated workflows

# Automated report generation
weekly_report = create_transaction_hypergraph('2024-10-01:2024-10-07')
print(f"Weekly report dataset: {weekly_report._dataset_id}")
```

## Benefits for Team Workflows

1. **API Integration**: dataset_id enables direct dashboard/API integration
2. **Automated Reporting**: Persist flag perfect for scheduled jobs
3. **Resource Efficiency**: Single server operation
4. **Consistent URLs**: dataset_id provides stable references for bookmarking

## Advanced Use Cases

**A. Multi-stage team pipeline**:
```python
# Data engineer: Creates base hypergraph
base_hg = g.gfql_remote([
    call('hypergraph', {'entity_types': ['user', 'merchant', 'product']}),
    call('save')
])

# Analyst: Adds layout and encoding
analyzed_hg = graphistry.from_dataset_id(base_hg._dataset_id).gfql_remote([
    call('fa2_layout', {'iterations': 1000}),
    call('encode_point_color', {'column': 'community'}),
    call('save')
])

# Manager: Views final result
final_url = analyzed_hg.url()
```

**B. A/B testing scenarios**:
```python
scenarios = ['scenario_a', 'scenario_b']
results = {}

for scenario in scenarios:
    result = g.gfql_remote([
        call('hypergraph', {'entity_types': get_entities(scenario)}),
        call('name', {'name': f'Test {scenario}'}),
        call('save')
    ])
    results[scenario] = result._dataset_id

# Compare results in dashboard
comparison_url = f"https://dashboard.com/compare?a={results['scenario_a']}&b={results['scenario_b']}"
```

## Technical Requirements

1. **Stable dataset_id generation**: Consistent across API calls
2. **Team permissions**: Respect organization sharing settings
3. **Metadata preservation**: Names, descriptions, encodings persist
4. **Efficient storage**: Reuse existing server-side dataset management