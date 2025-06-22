# Temporal Predicates in GFQL

PyGraphistry's GFQL (Graph Frame Query Language) provides comprehensive support for filtering graph data based on temporal (date/time) values. This guide covers how to use temporal predicates for datetime filtering in your graph queries.

## Overview

Temporal predicates allow you to filter nodes and edges based on datetime, date, or time values. All comparison predicates (`gt`, `lt`, `ge`, `le`, `eq`, `ne`, `between`) and the `is_in` predicate support temporal values.

## Basic Usage

### Datetime Filtering

Filter nodes or edges based on datetime values:

```python
import pandas as pd
from datetime import datetime
from graphistry import n
from graphistry.compute import gt, lt, between

# Filter nodes created after a specific datetime
recent_nodes = g.chain([
    n(filter_dict={"created_at": gt(pd.Timestamp("2023-01-01 12:00:00"))})
])

# Filter edges within a date range
date_range_edges = g.chain([
    n(edge_match={"timestamp": between(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31)
    )})
])
```

### Date-Only Filtering

For date comparisons (ignoring time):

```python
from datetime import date
from graphistry.compute import eq, ge

# Filter nodes by exact date
specific_date = g.chain([
    n(filter_dict={"event_date": eq(date(2023, 6, 15))})
])

# Filter nodes on or after a date
after_date = g.chain([
    n(filter_dict={"start_date": ge(date(2023, 1, 1))})
])
```

### Time-Only Filtering

Filter based on time of day:

```python
from datetime import time
from graphistry.compute import is_in, between

# Filter events at specific times
morning_events = g.chain([
    n(filter_dict={"event_time": is_in([
        time(9, 0, 0),
        time(9, 30, 0),
        time(10, 0, 0)
    ])})
])

# Filter events in time range
business_hours = g.chain([
    n(filter_dict={"timestamp": between(
        time(9, 0, 0),
        time(17, 0, 0)
    )})
])
```

## Timezone Support

Temporal predicates fully support timezone-aware datetime comparisons:

```python
import pytz

# Create timezone-aware timestamp
eastern = pytz.timezone('US/Eastern')
utc = pytz.UTC

# Filter with timezone-aware datetime
tz_aware_filter = g.chain([
    n(filter_dict={
        "timestamp": gt(pd.Timestamp("2023-01-01 12:00:00", tz=eastern))
    })
])

# Comparisons automatically handle timezone conversions
```

## Advanced Usage

### Mixed Temporal and Non-Temporal Predicates

Combine temporal predicates with other filters:

```python
from graphistry.compute import gt, lt, eq

# Complex filter with multiple conditions
complex_filter = g.chain([
    n(filter_dict={
        "created_at": gt(datetime(2023, 1, 1)),
        "status": eq("active"),
        "priority": gt(5)
    })
])
```

### Using Wire Protocol Dictionaries Directly

You can pass wire protocol dictionaries directly to predicates, which is useful for programmatic predicate creation or when working with JSON configurations:

```python
# Pass wire protocol dictionaries directly
filter_with_dict = g.chain([
    n(filter_dict={"timestamp": gt({
        "type": "datetime",
        "value": "2023-01-01T12:00:00",
        "timezone": "UTC"
    })})
])

# Works with all temporal predicates
date_range_filter = g.chain([
    n(filter_dict={"event_date": between(
        {"type": "date", "value": "2023-01-01"},
        {"type": "date", "value": "2023-12-31"}
    )})
])

# And with is_in for multiple values
time_filter = g.chain([
    n(filter_dict={"event_time": is_in([
        {"type": "time", "value": "09:00:00"},
        {"type": "time", "value": "12:00:00"},
        {"type": "time", "value": "17:00:00"}
    ])})
])
```

This is the same format used by the wire protocol, making it easy to:
- Store predicate configurations in JSON files
- Build predicates programmatically from external data sources
- Share predicate definitions between Python and other systems

### Temporal Predicates in Multi-Hop Queries

Use temporal filters in complex graph traversals:

```python
# Find all transactions after a date, then their related accounts
recent_transactions = g.chain([
    n(filter_dict={"type": eq("transaction"), 
                   "date": gt(date(2023, 6, 1))}),
    n(edge_match={"relationship": eq("involves")}),
    n(filter_dict={"type": eq("account")})
])
```

## Temporal Value Classes

PyGraphistry provides three temporal value classes for precise control:

### DateTimeValue

For full datetime with optional timezone:

```python
from graphistry.compute import DateTimeValue, gt

# Create datetime value with timezone
dt_value = DateTimeValue("2023-01-01T12:00:00", "US/Eastern")

# Use in predicate
filter_dt = g.chain([
    n(filter_dict={"timestamp": gt(dt_value)})
])
```

### DateValue

For date-only comparisons:

```python
from graphistry.compute import DateValue, between

# Create date values
start = DateValue("2023-01-01")
end = DateValue("2023-12-31")

# Use in between predicate
year_filter = g.chain([
    n(filter_dict={"event_date": between(start, end)})
])
```

### TimeValue

For time-of-day comparisons:

```python
from graphistry.compute import TimeValue, is_in

# Create time values
morning = TimeValue("09:00:00")
noon = TimeValue("12:00:00")

# Filter by specific times
time_filter = g.chain([
    n(filter_dict={"daily_event": is_in([morning, noon])})
])
```

## Best Practices

1. **Use Explicit Types**: Always use `pd.Timestamp`, `datetime`, `date`, or `time` objects instead of strings to avoid ambiguity.

2. **Timezone Awareness**: When working with timestamps across timezones, always specify timezones explicitly.

3. **Performance**: Temporal comparisons are optimized for pandas DataFrames. For large datasets, ensure your datetime columns are properly typed.

4. **JSON Serialization**: When serializing queries, temporal values are automatically converted to tagged dictionaries that preserve type and timezone information.

## Common Patterns

### Filter Recent Data

```python
from datetime import datetime, timedelta

# Get data from last 30 days
thirty_days_ago = datetime.now() - timedelta(days=30)
recent_data = g.chain([
    n(filter_dict={"timestamp": gt(pd.Timestamp(thirty_days_ago))})
])
```

### Business Hours Filtering

```python
# Filter events during business hours
business_hours = g.chain([
    n(filter_dict={
        "timestamp": between(time(9, 0, 0), time(17, 0, 0))
    })
])
```

### Quarterly Data Analysis

```python
# Q1 2023 data
q1_2023 = g.chain([
    n(filter_dict={
        "date": between(
            date(2023, 1, 1),
            date(2023, 3, 31)
        )
    })
])
```

## Error Handling

Temporal predicates include validation to prevent common errors:

```python
# This will raise an error - strings are ambiguous
# bad_filter = gt("2023-01-01")  # Don't do this

# Instead, be explicit
good_filter = gt(pd.Timestamp("2023-01-01"))  # Do this instead
```

## See Also

- [GFQL Predicates API Reference](../api/gfql/predicates.rst)
- [GFQL Chain Operations](./chain.md)
- [Temporal Predicates](../api/gfql/predicates.rst#temporal)