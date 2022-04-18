import torch
from util import norm_pdf
import matplotlib.pyplot as plt


def gauss_filter(ls, xs, ys, sigma):
    weights = norm_pdf(ls.reshape(len(ls), 1), xs.reshape(1, len(xs)), sigma)
    return torch.matmul(weights, ys) / weights.sum(axis=1)


def density(xs, ps, sigma):
    weights = norm_pdf(xs.reshape(len(xs), 1), ps.reshape(1, len(ps)), sigma)
    return weights.sum(axis=1) / len(ps)


def plot_loss(
    items,
    losses,
    approx=False,
    ylabel="Loss",
    ideal_losses=None,
    show_zero=True,
    log=False,
):

    lines = []
    ax = plt.axes()

    if log:
        losses = torch.log(losses)
        if ideal_losses != None:
            ideal_losses = torch.log(ideal_losses)

    if approx:
        linspace = torch.linspace(0, max(items), 500)
        max_items = items.max()
        # ax.plot(items, losses, 'lightskyblue', alpha=0.5)
        ax.plot(
            linspace,
            gauss_filter(linspace, items, losses, max_items / 500.0),
            "lightskyblue",
        )
        ax.plot(
            linspace,
            gauss_filter(linspace, items, losses, max_items / 150.0),
            "deepskyblue",
        )
        (loss_line,) = ax.plot(
            linspace,
            gauss_filter(linspace, items, losses, max_items / 50.0),
            "black",
            label=ylabel,
        )
        lines.append(loss_line)
    else:
        (loss_line,) = ax.plot(items, losses, "black", label=ylabel)
        lines.append(loss_line)

    if ideal_losses != None:
        ideal_items = items[ideal_losses > float("-inf")]
        ideal_losses = ideal_losses[ideal_losses > float("-inf")]
        (ideal_line,) = ax.plot(
            ideal_items, ideal_losses, "red", alpha=0.8, label="Ideal Loss"
        )
        lines.append(ideal_line)

    if show_zero:
        ax.fill_between(
            items,
            min(losses),
            max(losses),
            color="lime",
            where=(losses == 0),
            alpha=0.2,
        )

    plt.xlabel("Iterations")
    plt.ylabel(ylabel if len(lines) <= 1 else "Loss")
    # plt.ylim(0, max(losses)*1.03)
    if len(lines) > 1:
        ax.legend(handles=lines)

    plt.show()


def plot_bit_density(items):
    min_ = torch.squeeze(torch.min(items))
    max_ = torch.squeeze(torch.max(items))
    test_xs = torch.linspace(min_ - 0.1, max_ + 0.1, 500)
    plt.plot(test_xs, density(test_xs, items, 0.005), "deepskyblue")
    plt.plot(test_xs, density(test_xs, items, 0.01), "black")
    # means, _ = kmeans(items, 2)
    # plt.vlines(np.array(means).mean(), 0, 10, 'purple')
    plt.show()
