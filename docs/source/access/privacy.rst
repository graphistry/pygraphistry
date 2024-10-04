Privacy and Data Sharing in Graphistry
======================================

Graphistry provides powerful tools for visualizing and sharing graph data securely. Understanding how to manage privacy settings and share visualizations appropriately is essential for collaborative work and data security. This guide will help you understand how to control privacy settings using the Graphistry API. For more examples, see the `Sharing Tutorial Notebook <https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb>`_.

Overview of Privacy Settings
----------------------------

You have full control over who can view or edit your visualizations. By default, Graphistry visualizations are **public** but **unlisted**, meaning you need to have been given the secret ID of the visualization to know where it is, but do not need to log in to see it. Privacy settings can be adjusted when you create a plot using the `plot()` method.

Key privacy levels include:

- **Private**: Only you can view the visualization.
- **Organization (`"org"`)**: Anyone in your organization can view the visualization.
- **Public** (**unlisted**): Anyone with the link can view the visualization. Graphistry does not make the list of visualizations public, so this os the equivalent of the **unlisted** privacy mode in many platforms.
- **Custom Sharing**: Share with individual users (requires additional configuration).

When sharing with others, you may also configure settings such as `viewer` vs `editor`.

Getting Started with Privacy: Public (unlisted)
------------------------------------------------

Before adjusting privacy settings, ensure you have registered with Graphistry:

.. code-block:: python

    import graphistry

    graphistry.register(api=3, username='my_username', password='my_password')

By default, any plot you create is public (unlisted), meaning others will not know about your visualization, but if you share a link to it, they can see it without logging in.

Creating a Private Visualization
--------------------------------

You can set a visualization to a stricter mode by calling `graphistry.privacy()`:

.. code-block:: python

    graphistry.privacy()


    # Sample data
    edges = pd.DataFrame({
        'src': ['A', 'B', 'C'],
        'dst': ['B', 'C', 'A']
    })

    # Create a private plot
    plot_url = graphistry.edges(edges, 'src', 'dst').plot(render=False)

    print(f"Private visualization URL: {plot_url}")

If you are logged into your personal account, only you can access this plot. If you are logged into an organization, the visualization will be private to organization members. When anyone else obtains the URL, they won't be able to view it until you adjust the privacy settings. 

Sharing Visualizations Within Your Organization
-----------------------------------------------

To share a visualization with members of your organization:

.. code-block:: python

    graphistry.privacy(mode='organization')

    # Create an organization-shared plot
    plot_url = graphistry.edges(edges, 'src', 'dst').plot(render=False)

    print(f"Organization-shared visualization URL: {plot_url}")

Now, anyone within your organization who has access to Graphistry can view the plot using the provided URL.

Making Visualizations Public
----------------------------

To make a visualization accessible to anyone with the link:

.. code-block:: python

    graphistry.privacy(mode='public')

    # Create a public plot
    plot_url = graphistry.edges(edges, 'src', 'dst').plot(render=False)

    print(f"Public visualization URL: {plot_url}")

This setting is useful when sharing with external collaborators or embedding visualizations in public websites.

Controlling Edit Permissions
----------------------------

By default, shared visualizations are editable by same-org members. To allow others to edit or interact with the visualization settings, or set to read-only, you can reconfigure the policy:

.. code-block:: python

    VIEW = '10'
    EDIT = '20'
    graphistry.privacy(mode='organization', mode_action=EDIT)

    # Allow others to edit the plot
    plot_url = graphistry.edges(edges, 'src', 'dst').plot(render=False)

    print(f"Editable visualization URL: {plot_url}")


Understanding Privacy Levels
----------------------------

- **Private**: Only accessible to the creator.
- **Organization (`"org"`)**: Accessible to all users within your Graphistry organization.
- **Public**: Unlisted in any public index, but accessible to anyone with the link. Use cautiously, as this allows broad access.
- **Custom**: Advanced configurations for sharing with specific users.

Best Practices for Data Privacy
-------------------------------

- **Use Organization Sharing for Internal Collaboration**: Keeps data within your company's control.
- **Limit Public Sharing**: Only make visualizations public if the data is non-sensitive and intended for broad distribution.
- **Regularly Review Shared Visualizations**: Periodically check which visualizations are shared and adjust privacy settings as needed.
- **Use Secure Methods for Sharing Links**: When sharing URLs, use secure channels to prevent unauthorized access.

Advanced Features
------------------------------------------------------

Look at the documentation and tutorial for individual parameters for more advanced usage modes:

- Invite individual users, including with optional notification emails, using parameters `invited_users` and `notify`

- Use nested privacy settings (`g2 = g1.privacy()`)

Additional Resources
--------------------

For more detailed examples and advanced features, refer to the **Graphistry Sharing Tutorial** available in the official documentation or GitHub repository.

- **Sharing Tutorial Notebook**: `https://github.com/graphistry/pygraphistry/blob/master/demos/more_examples/graphistry_features/sharing_tutorial.ipynb`

This tutorial covers topics such as:

- Creating custom share links
- Embedding visualizations in web applications
- Using access tokens for secure sharing
- Advanced privacy configurations

Conclusion
----------

Managing privacy and sharing settings in Graphistry is straightforward and flexible. By understanding and utilizing these features, you can securely collaborate with others while maintaining control over your data.

Remember to:

- Choose the appropriate privacy level for your needs.
- Be cautious when making visualizations public.
- Regularly audit your shared visualizations.
- Use `graphistry.privacy()` to stay informed about your data handling.

