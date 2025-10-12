.. _gfql-predicates-quick:

GFQL Operator Reference
=======================

This reference outlines the operators available in GFQL for constructing predicates in your graph queries. These operators are wrappers around Pandas/cuDF functions, allowing you to express complex filtering conditions intuitively. See the API reference documentation for more details on individual operators.

Operators
---------

The following table lists the available operators, their descriptions, and examples of how to use them in GFQL.

**Numeric and Comparison Operators**

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``gt(value)``
     - Greater than ``value``.
     - ``n({ "age": gt(18) })``
   * - ``lt(value)``
     - Less than ``value``.
     - ``n({ "age": lt(65) })``
   * - ``ge(value)``
     - Greater than or equal to ``value``.
     - ``n({ "score": ge(90) })``
   * - ``le(value)``
     - Less than or equal to ``value``.
     - ``n({ "score": le(70) })``
   * - ``eq(value)``
     - Equal to ``value``.
     - ``n({ "status": eq("active") })``
   * - ``ne(value)``
     - Not equal to ``value``.
     - ``n({ "status": ne("inactive") })``
   * - ``between(lower, upper)``
     - Between ``lower`` and ``upper`` (inclusive).
     - ``n({ "age": between(18, 65) })``

.. note::
   All numeric comparison operators (``gt``, ``lt``, ``ge``, ``le``, ``eq``, ``ne``, ``between``) also support temporal values:
   
   - **DateTime**: ``n({ "created_at": gt(pd.Timestamp("2023-01-01 12:00:00")) })``
   - **Date**: ``n({ "event_date": eq(date(2023, 6, 15)) })``
   - **Time**: ``n({ "daily_time": between(time(9, 0), time(17, 0)) })``
   
   See :doc:`/gfql/datetime_filtering` for datetime filtering examples.

**Categorical Operators**

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``is_in(values)``
     - Value is in ``values`` list.
     - ``n({ "type": is_in(["person", "company"]) })``
   * - ``is_not_in(values)``
     - Value is not in ``values`` list.
     - ``n({ "type": is_not_in(["bot", "spam"]) })``
   * - ``duplicated(keep='first')``
     - Marks duplicated values.
     - ``n({ "email": duplicated() })``

**String Operators**

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``contains(pattern, case=True)``
     - String contains ``pattern``. Case-insensitive if ``case=False``.
     - ``n({ "name": contains("smith", case=False) })``
   * - ``startswith(prefix, case=True)``
     - String starts with ``prefix``. Case-insensitive if ``case=False``.
     - ``n({ "username": startswith("admin", case=False) })``
   * - ``endswith(suffix, case=True)``
     - String ends with ``suffix``. Case-insensitive if ``case=False``.
     - ``n({ "email": endswith(".com", case=False) })``
   * - ``match(pattern, case=True)``
     - String matches regex ``pattern`` from start. Case-insensitive if ``case=False``.
     - ``n({ "phone": match(r"^\d{3}-\d{4}$") })``
   * - ``fullmatch(pattern, case=True)``
     - String matches regex ``pattern`` entirely. Case-insensitive if ``case=False``.
     - ``n({ "code": fullmatch(r"\d{3}", case=False) })``
   * - ``isnumeric()``
     - String is numeric.
     - ``n({ "code": isnumeric() })``
   * - ``isalpha()``
     - String is alphabetic.
     - ``n({ "code": isalpha() })``
   * - ``isdigit()``
     - String is digit characters.
     - ``n({ "code": isdigit() })``
   * - ``islower()``
     - String is lowercase.
     - ``n({ "tag": islower() })``
   * - ``isupper()``
     - String is uppercase.
     - ``n({ "code": isupper() })``
   * - ``isspace()``
     - String contains only whitespace.
     - ``n({ "comment": isspace() })``
   * - ``isalnum()``
     - String is alphanumeric.
     - ``n({ "code": isalnum() })``
   * - ``isdecimal()``
     - String is decimal characters.
     - ``n({ "number": isdecimal() })``
   * - ``istitle()``
     - String is title-cased.
     - ``n({ "title": istitle() })``

**Null and NA Operators**

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``isna()``
     - Value is NA/NaN.
     - ``n({ "email": isna() })``
   * - ``notna()``
     - Value is not NA/NaN.
     - ``n({ "email": notna() })``
   * - ``isnull()``
     - Alias for ``isna()``.
     - ``n({ "email": isnull() })``
   * - ``notnull()``
     - Alias for ``notna()``.
     - ``n({ "email": notnull() })``

**Temporal Operators**

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
     - Example
   * - ``is_month_start()``
     - Date is the first day of the month.
     - ``n({ "date": is_month_start() })``
   * - ``is_month_end()``
     - Date is the last day of the month.
     - ``n({ "date": is_month_end() })``
   * - ``is_quarter_start()``
     - Date is the first day of the quarter.
     - ``n({ "date": is_quarter_start() })``
   * - ``is_quarter_end()``
     - Date is the last day of the quarter.
     - ``n({ "date": is_quarter_end() })``
   * - ``is_year_start()``
     - Date is the first day of the year.
     - ``n({ "date": is_year_start() })``
   * - ``is_year_end()``
     - Date is the last day of the year.
     - ``n({ "date": is_year_end() })``
   * - ``is_leap_year()``
     - Date is in a leap year.
     - ``n({ "date": is_leap_year() })``

Usage Examples
--------------

**Example 1: Filtering Nodes with Numeric Conditions**

.. code-block:: python

    from graphistry import n, gt, lt

    # Find nodes where age is greater than 18 and less than 30
    g_filtered = g.chain([
        n({ "age": gt(18) }),
        n({ "age": lt(30) })
    ])

**Example 2: Filtering Nodes by Category**

.. code-block:: python

    from graphistry import n, is_in

    # Find nodes of type 'person' or 'company'
    g_filtered = g.chain([
        n({ "type": is_in(["person", "company"]) })
    ])

**Example 3: Filtering Edges with String Conditions**

.. code-block:: python

    from graphistry import e_forward, contains

    # Find edges where the relation contains 'friend'
    g_filtered = g.chain([
        e_forward({ "relation": contains("friend") })
    ])

**Example 4: Combining Multiple Predicates**

.. code-block:: python

    from graphistry import n, eq, gt

    # Find 'person' nodes with age greater than 18
    g_filtered = g.chain([
        n({
            "type": eq("person"),
            "age": gt(18)
        })
    ])

Additional Notes
----------------

- **Lambda Functions**: You can use lambda functions for custom conditions.

  .. code-block:: python

      n({ "score": lambda x: (x > 50) & (x % 2 == 0) })

- **Importing Operators**: Remember to import the necessary functions.

  .. code-block:: python

      from graphistry import n, e_forward, gt, contains

- **Combining Conditions**: Use logical operators within lambdas for complex expressions.

  .. code-block:: python

      n({ "age": lambda x: (x > 18) & (x < 65) })

- **Predicates Module**: Operators are available in the `graphistry.predicates` module.

