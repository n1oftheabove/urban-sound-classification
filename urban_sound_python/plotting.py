import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_duration_and_count_of_sounds(df_meta):

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14,8), gridspec_kw={'height_ratios': [1,4]})

    # countplot on first row
    sns.countplot(data=df_meta,
                  x='class',
                  ax=ax[0])
    for p in ax[0].patches:
            ax[0].annotate('{:.0f}'.format(p.get_height()),
                           xy=(p.get_x() + p.get_width() / 2,
                               100),
                           ha='center',
                           va='center',
                           size=15,
                           color='white',
                           fontweight='bold')

    ax[0].xaxis.set_visible(False)
    ax[0].set_ylabel('Count of\n sounds', fontdict={"size":15})


    # boxplot & stripplot on second row
    sns.boxplot(data=df_meta,
                x='class',
                y='duration',
                showfliers=False,
                ax=ax[1],
                width=0.4,
               )

    sns.stripplot(x='class',
                  y='duration',
                  data=df_meta,
                  color="red",
                  jitter=0.2,
                  size=2.5,
                  ax=ax[1],
                 )
    plt.ylabel('duration in seconds', fontdict={"size":15})
    plt.xlabel('class', fontdict={"size":15})

    plt.yticks(size=13);
    plt.xticks(rotation=45, size=13);
    plt.tight_layout()
    # plt.title("Distribution of durations of soundfiles in the dataset per class", size=20)

def plot_duration_for_classes(df_meta):

    classes = df_meta['class'].unique()

    fig, axes = plt.subplots(5,2, figsize=(12,12))
    #plt.subplots_adjust(top=1.1)

    for ax, class_ in zip(axes.flatten(), classes):
        sns.histplot(data=df_meta[df_meta['class'] == class_],
                     x='duration',
                     bins=50,
                     ax=ax,
                     log_scale=(False,10),
                    )
        ax.set_title(class_, fontdict={'fontsize':20})
        ax.set_xlim(0, 4.5)
    plt.tight_layout()
