.. _kepler_preloaded_datasets:

Kepler.gl Preloaded Datasets
=================================

PyGraphistry provides preloaded Natural Earth geographic datasets for use with Kepler.gl visualizations.
These datasets include administrative boundaries at different levels with comprehensive attribute data.

Admin Region Hierarchy
----------------------

The Natural Earth data is organized into administrative levels:

* **0th Order (Countries)**: National boundaries - ``countries`` or ``zeroOrderAdminRegions``
* **1st Order (States/Provinces)**: Sub-national divisions - ``states``, ``provinces``, or ``firstOrderAdminRegions``

Countries Dataset (0th Order Admin Regions)
--------------------------------------------

The countries dataset contains 168 columns of data for each country. All column names are lowercase.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    from graphistry import KeplerDataset

    # Create a countries dataset
    countries_ds = KeplerDataset(
        type="countries",
        resolution=10,  # 10=high, 50=medium, 110=low
        include_countries=["United States of America", "Canada", "Mexico"]
    )

    # Get list of available columns
    columns = KeplerDataset.get_available_columns('countries')

Complete Column List with Example Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 168 columns with example values from United States (displayed in groups for clarity):

**Geographic and Administrative Columns**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``featurecla``
     - "Admin-0 country"
   * - ``scalerank``
     - 1
   * - ``labelrank``
     - 2
   * - ``sovereignt``
     - "United States of America"
   * - ``sov_a3``
     - "US1"
   * - ``adm0_dif``
     - 1
   * - ``level``
     - 2
   * - ``type``
     - "Country"
   * - ``tlc``
     - 1
   * - ``admin``
     - "United States of America"
   * - ``adm0_a3``
     - "USA"
   * - ``geou_dif``
     - 0
   * - ``geounit``
     - "United States of America"
   * - ``gu_a3``
     - "USA"
   * - ``su_dif``
     - 0
   * - ``subunit``
     - "United States"
   * - ``su_a3``
     - "USA"
   * - ``brk_diff``
     - 0

**Names and Identifiers**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``name``
     - "United States of America"
   * - ``name_long``
     - "United States"
   * - ``brk_a3``
     - "USA"
   * - ``brk_name``
     - "United States"
   * - ``brk_group``
     - ""
   * - ``abbrev``
     - "U.S.A."
   * - ``postal``
     - "US"
   * - ``formal_en``
     - "United States of America"
   * - ``formal_fr``
     - ""
   * - ``name_ciawf``
     - "United States"
   * - ``note_adm0``
     - ""
   * - ``note_brk``
     - ""
   * - ``name_sort``
     - "United States of America"
   * - ``name_alt``
     - ""

**Map Display Properties**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``mapcolor7``
     - 4
   * - ``mapcolor8``
     - 5
   * - ``mapcolor9``
     - 1
   * - ``mapcolor13``
     - 1
   * - ``min_zoom``
     - 0.0
   * - ``min_label``
     - 1.7
   * - ``max_label``
     - 5.7
   * - ``label_x``
     - -97.482602
   * - ``label_y``
     - 39.538479
   * - ``latitude``
     - 42.31380089200132
   * - ``longitude``
     - -105.33907490650022

**Demographics and Economics**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``pop_est``
     - 328239523.0
   * - ``pop_rank``
     - 17
   * - ``pop_year``
     - 2019
   * - ``gdp_md``
     - 21433226
   * - ``gdp_year``
     - 2019
   * - ``economy``
     - "1. Developed region: G7"
   * - ``income_grp``
     - "1. High income: OECD"

**ISO and International Codes**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``fips_10``
     - "US"
   * - ``iso_a2``
     - "US"
   * - ``iso_a2_eh``
     - "US"
   * - ``iso_a3``
     - "USA"
   * - ``iso_a3_eh``
     - "USA"
   * - ``iso_n3``
     - "840"
   * - ``iso_n3_eh``
     - "840"
   * - ``un_a3``
     - "840"
   * - ``wb_a2``
     - "US"
   * - ``wb_a3``
     - "USA"
   * - ``woe_id``
     - 23424977
   * - ``woe_id_eh``
     - 23424977
   * - ``woe_note``
     - "Exact WOE match as country"
   * - ``adm0_iso``
     - "USA"
   * - ``adm0_diff``
     - ""
   * - ``adm0_tlc``
     - "USA"

**Regional Classifications**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``continent``
     - "North America"
   * - ``region_un``
     - "Americas"
   * - ``subregion``
     - "Northern America"
   * - ``region_wb``
     - "North America"

**Country-Specific Admin Codes**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``adm0_a3_us``
     - "USA"
   * - ``adm0_a3_fr``
     - "USA"
   * - ``adm0_a3_ru``
     - "USA"
   * - ``adm0_a3_es``
     - "USA"
   * - ``adm0_a3_cn``
     - "USA"
   * - ``adm0_a3_tw``
     - "USA"
   * - ``adm0_a3_in``
     - "USA"
   * - ``adm0_a3_np``
     - "USA"
   * - ``adm0_a3_pk``
     - "USA"
   * - ``adm0_a3_de``
     - "USA"
   * - ``adm0_a3_gb``
     - "USA"
   * - ``adm0_a3_br``
     - "USA"
   * - ``adm0_a3_il``
     - "USA"
   * - ``adm0_a3_ps``
     - "USA"
   * - ``adm0_a3_sa``
     - "USA"
   * - ``adm0_a3_eg``
     - "USA"
   * - ``adm0_a3_ma``
     - "USA"
   * - ``adm0_a3_pt``
     - "USA"
   * - ``adm0_a3_ar``
     - "USA"
   * - ``adm0_a3_jp``
     - "USA"
   * - ``adm0_a3_ko``
     - "USA"
   * - ``adm0_a3_vn``
     - "USA"
   * - ``adm0_a3_tr``
     - "USA"
   * - ``adm0_a3_id``
     - "USA"
   * - ``adm0_a3_pl``
     - "USA"
   * - ``adm0_a3_gr``
     - "USA"
   * - ``adm0_a3_it``
     - "USA"
   * - ``adm0_a3_nl``
     - "USA"
   * - ``adm0_a3_se``
     - "USA"
   * - ``adm0_a3_bd``
     - "USA"
   * - ``adm0_a3_ua``
     - "USA"
   * - ``adm0_a3_un``
     - -99
   * - ``adm0_a3_wb``
     - -99

**Metadata Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``name_len``
     - 24
   * - ``long_len``
     - 13
   * - ``abbrev_len``
     - 6
   * - ``tiny``
     - -99
   * - ``homepart``
     - 1
   * - ``ne_id``
     - 1159321369
   * - ``wikidataid``
     - "Q30"

**Multilingual Names**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``name_ar``
     - "الولايات المتحدة"
   * - ``name_bn``
     - "মার্কিন যুক্তরাষ্ট্র"
   * - ``name_de``
     - "Vereinigte Staaten"
   * - ``name_en``
     - "United States of America"
   * - ``name_es``
     - "Estados Unidos"
   * - ``name_fa``
     - "ایالات متحده آمریکا"
   * - ``name_fr``
     - "États-Unis"
   * - ``name_el``
     - "Ηνωμένες Πολιτείες Αμερικής"
   * - ``name_he``
     - "ארצות הברית"
   * - ``name_hi``
     - "संयुक्त राज्य अमेरिका"
   * - ``name_hu``
     - "Amerikai Egyesült Államok"
   * - ``name_id``
     - "Amerika Serikat"
   * - ``name_it``
     - "Stati Uniti d'America"
   * - ``name_ja``
     - "アメリカ合衆国"
   * - ``name_ko``
     - "미국"
   * - ``name_nl``
     - "Verenigde Staten van Amerika"
   * - ``name_pl``
     - "Stany Zjednoczone"
   * - ``name_pt``
     - "Estados Unidos"
   * - ``name_ru``
     - "США"
   * - ``name_sv``
     - "USA"
   * - ``name_tr``
     - "Amerika Birleşik Devletleri"
   * - ``name_uk``
     - "Сполучені Штати Америки"
   * - ``name_ur``
     - "ریاستہائے متحدہ امریکا"
   * - ``name_vi``
     - "Hoa Kỳ"
   * - ``name_zh``
     - "美国"
   * - ``name_zht``
     - "美國"

**Feature Classification Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``fclass_iso``
     - "Admin-0 country"
   * - ``tlc_diff``
     - ""
   * - ``fclass_tlc``
     - "Admin-0 country"
   * - ``fclass_us``
     - ""
   * - ``fclass_fr``
     - ""
   * - ``fclass_ru``
     - ""
   * - ``fclass_es``
     - ""
   * - ``fclass_cn``
     - ""
   * - ``fclass_tw``
     - ""
   * - ``fclass_in``
     - ""
   * - ``fclass_np``
     - ""
   * - ``fclass_pk``
     - ""
   * - ``fclass_de``
     - ""
   * - ``fclass_gb``
     - ""
   * - ``fclass_br``
     - ""
   * - ``fclass_il``
     - ""
   * - ``fclass_ps``
     - ""
   * - ``fclass_sa``
     - ""
   * - ``fclass_eg``
     - ""
   * - ``fclass_ma``
     - ""
   * - ``fclass_pt``
     - ""
   * - ``fclass_ar``
     - ""
   * - ``fclass_jp``
     - ""
   * - ``fclass_ko``
     - ""
   * - ``fclass_vn``
     - ""
   * - ``fclass_tr``
     - ""
   * - ``fclass_id``
     - ""
   * - ``fclass_pl``
     - ""
   * - ``fclass_gr``
     - ""
   * - ``fclass_it``
     - ""
   * - ``fclass_nl``
     - ""
   * - ``fclass_se``
     - ""
   * - ``fclass_bd``
     - ""
   * - ``fclass_ua``
     - ""

**Geometry**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``_geometry``
     - MULTIPOLYGON(...)

States/Provinces Dataset (1st Order Admin Regions)
---------------------------------------------------

The states/provinces dataset contains administrative subdivisions for countries worldwide.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

    from graphistry import KeplerDataset

    # Create a states dataset for US states
    states_ds = KeplerDataset(
        type="states",
        include_countries=["United States of America"],
        include_1st_order_regions=["California", "Texas", "New York"]
    )

Complete Column List with Example Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All 115 columns with example values from California:

**Geographic and Administrative Columns**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``featurecla``
     - "Admin-1 states provinces"
   * - ``scalerank``
     - 2
   * - ``adm1_code``
     - "USA-3521"
   * - ``diss_me``
     - 3521
   * - ``iso_3166_2``
     - "US-CA"
   * - ``wikipedia``
     - "http://en.wikipedia.org/wiki/California"
   * - ``iso_a2``
     - "US"
   * - ``adm0_sr``
     - 8

**Names and Identifiers**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``name``
     - "California"
   * - ``name_alt``
     - "CA|Calif."
   * - ``name_local``
     - ""
   * - ``type``
     - "State"
   * - ``type_en``
     - "State"
   * - ``code_local``
     - "US06"
   * - ``code_hasc``
     - "US.CA"
   * - ``note``
     - ""
   * - ``hasc_maybe``
     - ""

**Regional Information**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``region``
     - "West"
   * - ``region_cod``
     - ""
   * - ``region_sub``
     - "Pacific"
   * - ``sub_code``
     - ""
   * - ``provnum_ne``
     - 0.0
   * - ``gadm_level``
     - 1

**Administrative Details**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``check_me``
     - 20
   * - ``datarank``
     - 1
   * - ``abbrev``
     - "Calif."
   * - ``postal``
     - "CA"
   * - ``area_sqkm``
     - 0.0
   * - ``sameascity``
     - -99
   * - ``labelrank``
     - 0
   * - ``name_len``
     - 10

**Map Display Properties**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``mapcolor9``
     - 1
   * - ``mapcolor13``
     - 1
   * - ``min_label``
     - 3.5
   * - ``max_label``
     - 7.5
   * - ``min_zoom``
     - 2.0

**External References**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``fips``
     - "US06"
   * - ``fips_alt``
     - ""
   * - ``woe_id``
     - 2347563.0
   * - ``woe_label``
     - "California, US, United States"
   * - ``woe_name``
     - "California"
   * - ``wikidataid``
     - "Q99"
   * - ``ne_id``
     - 1159308415

**Geographic Coordinates**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``latitude``
     - 37.1259483770762
   * - ``longitude``
     - -119.44202946142391

**Parent Country Information**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``sov_a3``
     - "US1"
   * - ``adm0_a3``
     - "USA"
   * - ``adm0_label``
     - 2
   * - ``admin``
     - "United States of America"
   * - ``geonunit``
     - "United States of America"
   * - ``gu_a3``
     - "USA"

**GeoNames Integration**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``gn_id``
     - 5332921.0
   * - ``gn_name``
     - "California"
   * - ``gns_id``
     - -1.0
   * - ``gns_name``
     - ""
   * - ``gn_level``
     - 1.0
   * - ``gn_region``
     - ""
   * - ``gn_a1_code``
     - "US.CA"
   * - ``gns_level``
     - -1.0
   * - ``gns_lang``
     - ""
   * - ``gns_adm1``
     - ""
   * - ``gns_region``
     - ""

**Multilingual Names**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``name_ar``
     - "كاليفورنيا"
   * - ``name_bn``
     - "ক্যালিফোর্নিয়া"
   * - ``name_de``
     - "Kalifornien"
   * - ``name_en``
     - "California"
   * - ``name_es``
     - "California"
   * - ``name_fr``
     - "Californie"
   * - ``name_el``
     - "Καλιφόρνια"
   * - ``name_hi``
     - "कैलिफ़ोर्निया"
   * - ``name_hu``
     - "Kalifornia"
   * - ``name_id``
     - "California"
   * - ``name_it``
     - "California"
   * - ``name_ja``
     - "カリフォルニア州"
   * - ``name_ko``
     - "캘리포니아"
   * - ``name_nl``
     - "Californië"
   * - ``name_pl``
     - "Kalifornia"
   * - ``name_pt``
     - "Califórnia"
   * - ``name_ru``
     - "Калифорния"
   * - ``name_sv``
     - "Kalifornien"
   * - ``name_tr``
     - "Kaliforniya"
   * - ``name_vi``
     - "California"
   * - ``name_zh``
     - "加利福尼亚州"
   * - ``name_he``
     - "קליפורניה"
   * - ``name_uk``
     - "Каліфорнія"
   * - ``name_ur``
     - "کیلی فورنیا"
   * - ``name_fa``
     - "کالیفرنیا"
   * - ``name_zht``
     - "加利福尼亞州"

**Feature Classification Fields**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``fclass_iso`` to ``fclass_tlc``
     - "" (empty for all)

**Geometry**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Example Value
   * - ``_geometry``
     - MULTIPOLYGON(...)

Data Source
-----------

These datasets are derived from Natural Earth (https://www.naturalearthdata.com/), a public domain map dataset available at 1:10m, 1:50m, and 1:110m scales. The data is updated periodically to reflect political and demographic changes.

See Also
--------

* :doc:`/api/plotter` - Main plotting interface with Kepler support
* Natural Earth documentation: https://www.naturalearthdata.com/