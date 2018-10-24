plot_imshow(train_dataset, 'SENTIMENT', 'docs')
plt.title('Sentiment by Geo-Groupby Documents')

plot_imshow(train_dataset['X'][~train_dataset['Y']], example_topic_column_name)
plt.title('LDA Topic {} by Geo-Groupby
Documents'.format(example_topic_column_name))

plot_imshow(threat_datasets['KDE']['df'], 'KDE')
plt.title('Threat for each cell in Chicago by KDE model')

plot_imshow(threat_datasets['KDE+LDA']['df'], 'logreg')
plt.title('Threat for each cell in Chicago by KDE+LDA model')

plot_imshow(threat_datasets['KDE+SENTIMENT']['df'], 'logreg')
plt.title('Threat for each cell in Chicago by KDE+SENTIMENT model')

plot_imshow(threat_datasets['KDE+SENTIMENT+LDA']['df'], 'logreg')
plt.title('Threat for each cell in Chicago by KDE+SENTIMENT+LDA model')
