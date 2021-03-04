---
name: Bug report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Please provide code and data than can be run without editing:

```python
import pandas as pd
import graphistry
#graphistry.register(api=3, username='...', password='...')

graphistry.edges(pd.from_csv('https://data.csv'), 's', 'd')).plot()
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Actual behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Browser environment (please complete the following information):**
 - OS: [e.g. iOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**PyGraphistry environment**
 - Version [e.g. 0.14.0, print via `graphistry.__version__`]
 - Python Version [e.g. Python 3.7.7] 
 - Where run [e.g., Graphistry 2.35.9 Jupyter]

**Additional context**
Add any other context about the problem here.
