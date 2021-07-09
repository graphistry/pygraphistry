---
name: Bug report
about: Create a report to help us improve
title: "[BUG] "
labels: bug
assignees: ''

---

**Describe the bug**
1-3 sentences is fine 💪

**To Reproduce**
Code, including data, than can be run without editing:

```python
import pandas as pd
import graphistry
#graphistry.register(api=3, username='...', password='...')

graphistry.edges(pd.from_csv('https://data.csv'), 's', 'd')).plot()
```

**Expected behavior**
What should have happened

**Actual behavior**
What did happen

**Screenshots**
If applicable, any screenshots to help explain the issue

**Browser environment (please complete the following information):**
 - OS: [e.g. iOS]
 - Browser [e.g. chrome, safari]
 - Version [e.g. 22]

**Graphistry GPU server environment**
 - Where run [e.g., Hub, AWS, on-prem]
 - If self-hosting, Graphistry Version [e.g. 0.14.0, see bottom of a viz or login dashboard]
- If self-hosting, any OS/GPU/driver versions

**PyGraphistry API client environment**
 - Where run [e.g., Graphistry 2.35.9 Jupyter]
 - Version [e.g. 0.14.0, print via `graphistry.__version__`]
 - Python Version [e.g. Python 3.7.7] 

**Additional context**
Add any other context about the problem here.
