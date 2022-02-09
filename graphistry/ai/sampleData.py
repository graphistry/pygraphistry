import numpy as np
import torch
import dgl


def get_heterograph():
    # from https://docs.dgl.ai/en/0.6.x/guide/training.html#guide-training-heterogeneous-graph-example
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10

    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)

    hetero_graph = dgl.heterograph(
        {
            ("user", "follow", "user"): (follow_src, follow_dst),
            ("user", "followed-by", "user"): (follow_dst, follow_src),
            ("user", "click", "item"): (click_src, click_dst),
            ("item", "clicked-by", "user"): (click_dst, click_src),
            ("user", "dislike", "item"): (dislike_src, dislike_dst),
            ("item", "disliked-by", "user"): (dislike_dst, dislike_src),
        }
    )

    hetero_graph.nodes["user"].data["feature"] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes["item"].data["feature"] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes["user"].data["label"] = torch.randint(
        0, n_user_classes, (n_users,)
    )
    hetero_graph.edges["click"].data["label"] = torch.randint(
        1, n_max_clicks, (n_clicks,)
    ).float()

    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes["user"].data["train_mask"] = torch.zeros(
        n_users, dtype=torch.bool
    ).bernoulli(0.6)
    hetero_graph.edges["click"].data["train_mask"] = torch.zeros(
        n_clicks, dtype=torch.bool
    ).bernoulli(0.6)
    # todo add print out
    return hetero_graph
