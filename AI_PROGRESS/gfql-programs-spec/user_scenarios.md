# User Scenarios for GFQL Programs

## Alex - Security Analyst Scenarios

### Scenario A1: Multi-Source Threat Investigation
**Goal**: Correlate alerts from IDS with network traffic and user behavior
**Features**: RemoteGraph, GraphUnion, Chain operations
**Challenge**: Different schemas across sources, large data volumes

### Scenario A2: Lateral Movement Detection
**Goal**: Track potential lateral movement across systems over time
**Features**: Complex DAG with multiple hops, dotted references
**Challenge**: Resource limits on deep traversals, timeout handling

### Scenario A3: Incident Timeline Reconstruction
**Goal**: Build complete timeline merging logs, alerts, and network data
**Features**: GraphUnion with temporal ordering, call operations for layout
**Challenge**: Handling missing data, performance with large timespans

### Scenario A4: Threat Hunting Workflow
**Goal**: Create reusable hunting patterns for specific TTPs
**Features**: DAG templates, parameterized remote graphs
**Challenge**: Making workflows shareable, version control

## Sam - Data Scientist Scenarios

### Scenario S1: Fraud Ring Detection Pipeline
**Goal**: Build end-to-end pipeline from transactions to fraud clusters
**Features**: Complex DAG, UMAP/clustering calls, graph combinators
**Challenge**: Memory limits with large transaction graphs

### Scenario S2: Feature Engineering for ML
**Goal**: Extract graph features for downstream ML models
**Features**: Call operations (centrality, embeddings), DAG composition
**Challenge**: Handling failures in long-running pipelines

### Scenario S3: A/B Testing Graph Algorithms
**Goal**: Compare different clustering approaches on same data
**Features**: Parallel DAG branches, call operations with parameters
**Challenge**: Resource allocation across parallel operations

### Scenario S4: Real-time Fraud Scoring
**Goal**: Score new transactions against historical patterns
**Features**: RemoteGraph for history, GraphIntersect for matching
**Challenge**: Latency requirements, caching strategies

## Jordan - Business Analyst Scenarios

### Scenario J1: Department Collaboration Report
**Goal**: Show interactions between departments from email/meeting data
**Features**: Simple RemoteGraph loading, GraphUnion, layout
**Challenge**: Understanding error messages, handling auth failures

### Scenario J2: Customer Journey Analysis
**Goal**: Combine touchpoints from marketing, sales, support
**Features**: GraphUnion with merge policies, basic filtering
**Challenge**: Dealing with duplicate customer IDs

### Scenario J3: Quarterly Comparison Dashboard
**Goal**: Compare network patterns between quarters
**Features**: Multiple RemoteGraphs, GraphSubtract to show changes
**Challenge**: Resource limits on large quarterly datasets

## Morgan - DevOps Engineer Scenarios

### Scenario M1: Service Dependency Mapping
**Goal**: Visualize microservice dependencies with health status
**Features**: Nested DAGs for service groups, call for layout
**Challenge**: Deep nesting with dotted references

### Scenario M2: Incident Impact Analysis
**Goal**: Find all services affected by infrastructure failure
**Features**: Multi-hop traversal, GraphIntersect with alert data
**Challenge**: Timeout handling for large service meshes

### Scenario M3: Capacity Planning Analysis
**Goal**: Analyze resource usage patterns across services
**Features**: Call operations for metrics, graph combinators
**Challenge**: Setting appropriate resource quotas

### Scenario M4: Configuration Drift Detection
**Goal**: Compare actual vs intended infrastructure state
**Features**: RemoteGraphs for both states, GraphSubtract
**Challenge**: Handling large configuration graphs efficiently

## Casey - Compliance Officer Scenarios

### Scenario C1: Sanctions Screening
**Goal**: Check entities against multiple sanctions lists
**Features**: RemoteGraphs for lists, GraphIntersect
**Challenge**: Handling fuzzy matching requirements

### Scenario C2: Beneficial Ownership Mapping
**Goal**: Trace ultimate beneficial ownership through entities
**Features**: Multi-hop traversal with filters, depth limits
**Challenge**: Circular ownership structures

### Scenario C3: Regulatory Change Impact
**Goal**: Identify affected entities from new regulations
**Features**: GraphUnion of entity types, filtered operations
**Challenge**: Clear audit trail requirements

## Riley - Research Scientist Scenarios

### Scenario R1: Protein Interaction Analysis
**Goal**: Analyze protein-protein interaction networks
**Features**: Custom call operations, large graph handling
**Challenge**: Memory limits with genome-scale networks

### Scenario R2: Multi-Omics Integration
**Goal**: Combine genomic, proteomic, metabolomic networks
**Features**: Complex GraphUnion with schema mapping
**Challenge**: Handling heterogeneous data types

### Scenario R3: Pathway Enrichment Pipeline
**Goal**: Run enrichment analysis across pathways
**Features**: Parallel DAG operations, custom algorithms
**Challenge**: Computational resource allocation

### Scenario R4: Comparative Network Analysis
**Goal**: Compare networks across species/conditions
**Features**: Multiple RemoteGraphs, graph difference operations
**Challenge**: Dealing with incomplete/missing data

## Taylor - Supply Chain Analyst Scenarios

### Scenario T1: Multi-Tier Risk Assessment
**Goal**: Assess risk propagation through 4+ tiers of suppliers
**Features**: Deep traversal, risk scoring algorithms, web dashboard
**Challenge**: Handling circular supplier relationships, real-time updates

### Scenario T2: Disruption Simulation
**Goal**: Simulate impact of supplier failures on production
**Features**: What-if analysis, GraphSubtract for failure scenarios
**Challenge**: Complex propagation rules, performance with large networks

### Scenario T3: Supplier Diversity Analysis
**Goal**: Identify single points of failure in supply chain
**Features**: Centrality analysis, critical path identification
**Challenge**: Integration with ERP data, executive visualization needs

## Quinn - Cyber Threat Hunter Scenarios

### Scenario Q1: Behavioral Anomaly Hunting
**Goal**: Find unusual patterns in user/system behavior
**Features**: Louie conversational queries, anomaly detection, pattern matching
**Challenge**: Reducing false positives, iterative hypothesis testing

### Scenario Q2: Living Off the Land Detection
**Goal**: Detect threats using legitimate tools
**Features**: Complex behavioral patterns, time-based analysis
**Challenge**: Distinguishing legitimate from malicious use

### Scenario Q3: Data Exfiltration Hunting
**Goal**: Identify potential data exfiltration patterns
**Features**: Network flow analysis, statistical anomalies
**Challenge**: Large data volumes, encryption blind spots

## Harper - Cyber Threat Intelligence Scenarios

### Scenario H1: IOC Correlation and Attribution
**Goal**: Correlate IOCs across feeds and attribute to threat actors
**Features**: Multi-source integration, MITRE mapping, confidence scoring
**Challenge**: Conflicting attribution, data freshness

### Scenario H2: Campaign Tracking
**Goal**: Track evolution of threat campaigns over time
**Features**: Temporal analysis, pattern evolution, API development
**Challenge**: Maintaining historical context, sharing with partners

### Scenario H3: Threat Intelligence API
**Goal**: Build API for internal teams to query threat data
**Features**: API wrapper around GFQL queries, caching, rate limiting
**Challenge**: Performance requirements, access control

## Blake - Tier 1 SOC Analyst Scenarios

### Scenario B1: Alert Triage Workflow
**Goal**: Quickly triage high volume of security alerts
**Features**: Template queries, Louie assistance, simple visualizations
**Challenge**: Alert fatigue, limited technical skills

### Scenario B2: Initial Investigation
**Goal**: Perform first-level investigation of security incidents
**Features**: Guided workflows, pre-built queries, escalation criteria
**Challenge**: Knowing when to escalate, documentation requirements

### Scenario B3: False Positive Identification
**Goal**: Identify and document false positive patterns
**Features**: Pattern templates, simple filtering, tagging
**Challenge**: Building confidence in decisions, reducing repeat alerts

## Dakota - Tier 2 SOC Analyst Scenarios

### Scenario D1: Deep Dive Investigation
**Goal**: Perform thorough investigation of escalated incidents
**Features**: Complex multi-stage queries, custom algorithms, automation
**Challenge**: Efficiency under pressure, comprehensive documentation

### Scenario D2: Threat Remediation Planning
**Goal**: Develop remediation plans for confirmed threats
**Features**: Impact analysis, dependency mapping, playbook creation
**Challenge**: Understanding business impact, coordinating response

### Scenario D3: Investigation Automation
**Goal**: Automate repetitive investigation tasks
**Features**: Query templates, workflow automation, dashboard creation
**Challenge**: Balancing automation with human judgment