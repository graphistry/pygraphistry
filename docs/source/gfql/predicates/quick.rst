.. _gfql-predicates-quick:

GFQL Operator Reference
=======================

This reference outlines the operators available in GFQL for constructing predicates in your graph queries. These operators are wrappers around Pandas/cuDF functions, allowing you to express complex filtering conditions intuitively. See the API reference documentation for more details on individual operators.

Operators
---------

The following table lists the available operators, their descriptions, and examples of how to use them in GFQL.

WHERE Operators (Cross-Reference)
---------------------------------

This page covers predicate functions used inside step filters like
``n({...})`` and ``e_forward({...})``. WHERE operators are documented separately:

- Same-path MATCH WHERE uses ``compare(col(...), op, col(...))`` with
  ``op`` in ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``.
- Row-pipeline WHERE uses ``where_rows(expr="...")`` with comparators
  ``=``, ``!=``, ``<>``, ``<``, ``<=``, ``>``, ``>=``.

See :doc:`/gfql/where` (same-path constraints) and :doc:`/gfql/return`
(``MATCH ... RETURN`` row pipelines).

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
     - ``n({ "status": eq("active") })`` (supports strings, numeric, temporals; use ``isna()/notna()`` for nulls)
   * - ``ne(value)``
     - Not equal to ``value``.
     - ``n({ "status": ne("inactive") })``
   * - ``between(lower, upper)``
     - Between ``lower`` and ``upper`` (inclusive).
     - ``n({ "age": between(18, 65) })``

.. note::
   Null handling and temporal comparisons are separate:

   - Use ``isna()`` / ``notna()`` for null-safe checks:
     - ``n({ "closed_at": isna() })``
     - ``n({ "created_at": notna() })``
   - Use comparison operators for non-null values (including temporal columns):
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
     - Value is not in ``values`` list. *(Not yet implemented.)*
     - Use ``n(query="type not in ['bot', 'spam']")``
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

.. doc-test: skip

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

**Example 5: Same-Path Constraint with WHERE**

.. code-block:: python

    from graphistry import n, e_forward, col, compare

    g_filtered = g.gfql(
        [
            n({"type": "account"}, name="a"),
            e_forward(),
            n({"type": "user"}, name="c"),
        ],
        where=[compare(col("a", "owner_id"), "==", col("c", "owner_id"))],
    )

Additional Notes
----------------

- **Predicate Functions**: Use predicate instances for filter conditions.

  .. code-block:: python

      n({ "score": gt(50) })

  For compound conditions (e.g., ``score > 50 AND score is even``), use a
  ``query`` string instead:

  .. code-block:: python

      n(query="score > 50 and score % 2 == 0")

  .. note::

     Lambda functions in ``filter_dict`` (e.g., ``n({"score": lambda x: ...})``)
     are no longer supported because ``filter_dict`` values must be
     JSON-serializable for the wire protocol and remote execution. Use
     predicates like ``gt()``, ``between()``, or ``query=`` strings for
     compound conditions.

- **Importing Operators**: Remember to import the necessary functions.

  .. code-block:: python

      from graphistry import n, e_forward, gt, contains

- **Combining Conditions**: Use range predicates or ``query`` strings for complex expressions.

  .. code-block:: python

      # Range predicate
      n({ "age": between(19, 64) })

      # Or equivalently with a query string
      n(query="age > 18 and age < 65")

- **Predicates Module**: Operators are available in the `graphistry.predicates` module.
