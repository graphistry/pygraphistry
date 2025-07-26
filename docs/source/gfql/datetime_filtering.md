# Working with Dates and Times

GFQL predicates support filtering by datetime, date, and time values. This guide covers common patterns and gotchas when working with temporal data.

## Required Imports

```python
# Core imports
import graphistry
from graphistry import n, e_forward, e_reverse, e_undirected

# Temporal predicates
from graphistry.compute import (
    gt, lt, ge, le, eq, ne, between, is_in,
    DateTimeValue, DateValue, TimeValue
)

# Standard datetime types
import pandas as pd
from datetime import datetime, date, time, timedelta
import pytz  # For timezone support
```

## Supported Types and Standards

### Supported Python Types

- [`pd.Timestamp`](https://pandas.pydata.org/docs/reference/api/pandas.Timestamp.html) - Pandas timestamp
- [`datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime) - Python datetime
- [`date`](https://docs.python.org/3/library/datetime.html#datetime.date) - Date only (no time)
- [`time`](https://docs.python.org/3/library/datetime.html#datetime.time) - Time only (no date)
- Wire protocol dicts - For ISO strings and JSON compatibility

```python
# Use datetime objects
gt(pd.Timestamp("2023-01-01 12:00:00"))
between(datetime(2023, 1, 1), datetime(2023, 12, 31))

# Wire protocol dicts accept ISO strings
gt({"type": "datetime", "value": "2023-01-01T00:00:00", "timezone": "UTC"})

# Raw strings raise ValueError
gt("2023-01-01")  # ValueError
```

### Creating from Strings

```python
# Timestamps
pd.Timestamp("2023-01-01T12:00:00Z")  # UTC
pd.Timestamp("2023-01-01 12:00:00")   # Naive

# Date/Time objects
date.fromisoformat("2023-01-01")      # date(2023, 1, 1)
time.fromisoformat("14:30:00")        # time(14, 30, 0)
```

### Standards
- [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime strings: `"2023-01-01T12:00:00Z"`
- [IANA timezone](https://www.iana.org/time-zones) names: `"US/Eastern"`, `"UTC"`

### Wire Protocol Types
For JSON serialization and cross-system compatibility:
- **DateTimeWire**: `{"type": "datetime", "value": "ISO-8601-string", "timezone": "IANA-timezone"}`
- **DateWire**: `{"type": "date", "value": "YYYY-MM-DD"}`
- **TimeWire**: `{"type": "time", "value": "HH:MM:SS[.ffffff]"}`

Note: The `timezone` field is optional for DateTimeWire and defaults to "UTC" if omitted.

## Basic Usage

### Datetime Filtering

Filter nodes or edges based on datetime values:

```python
import pandas as pd
from datetime import datetime
from graphistry import n
from graphistry.compute import gt, lt, between

# Filter nodes created after a specific datetime
recent_nodes = g.gfql([
    n(filter_dict={"created_at": gt(pd.Timestamp("2023-01-01 12:00:00"))})
])

# Filter edges within a date range
date_range_edges = g.gfql([
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
specific_date = g.gfql([
    n(filter_dict={"event_date": eq(date(2023, 6, 15))})
])

# Filter nodes on or after a date
after_date = g.gfql([
    n(filter_dict={"start_date": ge(date(2023, 1, 1))})
])
```

### Time-Only Filtering

Filter based on time of day:

```python
from datetime import time
from graphistry.compute import is_in, between

# Filter events at specific times
morning_events = g.gfql([
    n(filter_dict={"event_time": is_in([
        time(9, 0, 0),
        time(9, 30, 0),
        time(10, 0, 0)
    ])})
])

# Filter events in time range
business_hours = g.gfql([
    n(filter_dict={"timestamp": between(
        time(9, 0, 0),
        time(17, 0, 0)
    )})
])
```

## Timezone Support

```python
import pytz

# Timezone-aware filtering
eastern = pytz.timezone('US/Eastern')
tz_aware_filter = g.gfql([
    n(filter_dict={
        "timestamp": gt(pd.Timestamp("2023-01-01 12:00:00", tz=eastern))
    })
])
```

Comparisons automatically handle timezone conversions.

## Advanced Usage

### Mixed Temporal and Non-Temporal Predicates

Combine temporal predicates with other filters:

```python
from graphistry.compute import gt, lt, eq

# Complex filter with multiple conditions
complex_filter = g.gfql([
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
filter_with_dict = g.gfql([
    n(filter_dict={"timestamp": gt({
        "type": "datetime",
        "value": "2023-01-01T12:00:00",
        "timezone": "UTC"
    })})
])

# Works with all temporal predicates
date_range_filter = g.gfql([
    n(filter_dict={"event_date": between(
        {"type": "date", "value": "2023-01-01"},
        {"type": "date", "value": "2023-12-31"}
    )})
])

# And with is_in for multiple values
time_filter = g.gfql([
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
recent_transactions = g.gfql([
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
filter_dt = g.gfql([
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
year_filter = g.gfql([
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
time_filter = g.gfql([
    n(filter_dict={"daily_event": is_in([morning, noon])})
])
```

## Best Practices

1. **Use Explicit Types**: Always use `pd.Timestamp`, `datetime`, `date`, or `time` objects instead of strings to avoid ambiguity.

2. **Timezone Awareness**: When working with timestamps across timezones, always specify timezones explicitly.

3. **Performance**: Temporal comparisons are optimized for pandas DataFrames. For large datasets, ensure your datetime columns are properly typed.

4. **JSON Serialization**: When serializing queries, temporal values are automatically converted to tagged dictionaries that preserve type and timezone information.

## Unsupported Features

### Duration/Interval Support
Currently, PyGraphistry does not support duration or interval types (e.g., ISO 8601 durations like "P1D" or "PT2H"). For duration-based queries:

```python
# Instead of duration literals, calculate explicit timestamps
from datetime import datetime, timedelta

# Find events within last 7 days
now = datetime.now()
week_ago = now - timedelta(days=7)
recent_events = g.gfql([
    n(filter_dict={"timestamp": gt(pd.Timestamp(week_ago))})
])

# For recurring intervals, use multiple conditions
business_days = g.gfql([
    n(filter_dict={
        "timestamp": between(
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-12-31")
        )
    })
])
```

## Common Patterns

### Filter Recent Data

```python
from datetime import datetime, timedelta

# Get data from last 30 days
thirty_days_ago = datetime.now() - timedelta(days=30)
recent_data = g.gfql([
    n(filter_dict={"timestamp": gt(pd.Timestamp(thirty_days_ago))})
])
```

### Business Hours Filtering

```python
# Filter events during business hours
business_hours = g.gfql([
    n(filter_dict={
        "timestamp": between(time(9, 0, 0), time(17, 0, 0))
    })
])
```

### Quarterly Data Analysis

```python
# Q1 2023 data
q1_2023 = g.gfql([
    n(filter_dict={
        "date": between(
            date(2023, 1, 1),
            date(2023, 3, 31)
        )
    })
])
```

## Error Handling

```python
# Strings raise ValueError - always use datetime objects
gt("2023-01-01")                # ValueError: Raw string not allowed
gt(pd.Timestamp("2023-01-01"))  # Correct: Use pandas Timestamp
```

## See Also

- [GFQL Predicates API Reference](../api/gfql/predicates.rst)
- [GFQL Chain Operations](../api/gfql/chain.rst)
- [Wire Protocol Reference](wire_protocol_examples.md)