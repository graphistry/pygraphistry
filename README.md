# Graphistry in a Nutshell

### Installation

The simplest way is to use pip:

```
$ pip install graphistry
```

## First Graph

We recommend [Pandas](http://pandas.pydata.org) to load and process data. If you don't have Pandas already, install it with `$ pip install pandas`.

#### Loading data
Let's load the characters from [Les Miserables](http://en.wikipedia.org/wiki/Les_Misérables). Our  [dataset is a CSV file](http://gist.github.com/thibaudh/3da4096c804680f549e6/) that looks like this:

| source        | target        | value  |
| ------------- |:-------------:| ------:|
| Cravatte |	Myriel | 1| Valjean	| Mme.Magloire | 3| Valjean	| Mlle.Baptistine | 3

*Source* and *target* are characters name, and the *value* field count the number of time they meet. Loadin the data is easy with Pandas:

```python
import pandas as pd

links = pd.read_csv('./lesmiserables.csv')
```

#### Visualize Data
The graphistry package can plot graphs directly from Pandas dataframes. We do specify the name of the two columns indicating the start and end nodes of each edges with *sourcefield* and *destfield*. 

```python
import graphistry as g

g.plot(links, sourcefield="source", destfield="target")
```

You should see a beautiful graph like this one:
![Graph of Miserables](http://i.imgur.com/N6UZLSG.png)

Note that the visualization is computed in the cloud. As such an Internet connection is required.




