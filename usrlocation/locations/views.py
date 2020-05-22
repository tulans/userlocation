import matplotlib
import pandas as pd
from django.shortcuts import render
from sklearn.cluster import KMeans

# import io
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from django.http import HttpResponse
# from matplotlib.backends.backend_agg import FigureCanvasAgg

matplotlib.use('Agg')
plt = matplotlib.pyplot

def plot_graph(request):
    csv_file = "locations/CountriesExerciseCluster.csv"
    data = pd.read_csv(csv_file)
    print(data.head())
    x = data.iloc[:, 1:2]
    wcss = []
    for i in range(1, 15):
        km = KMeans(i)
        km.fit(x)
        wcss_iter = km.inertia_
        wcss.append(wcss_iter)
    number_clusters = range(1, 15)
    plt.plot(number_clusters, wcss)
    # fig, ax = plt.subplots()
    plt.title('Elbow method')
    plt.xlabel('Numbeer of clusters')
    plt.ylabel('within cluster sum of squares')
    # plt.show()

    # Let's take Lat, Long and Language to perform cluster analysis
    k = 10
    kmeans = KMeans(k)
    print('Trying ' + str(k) + ' means cluster ')
    identified_cluster = kmeans.fit_predict(x)
    data['Cluster'] = identified_cluster
    plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    # plt.show()


    # --display only image --
    # FigureCanvasAgg(fig)
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # byteImg = Image.open("../static/usrlocation/images/g1.png")
    # byteImg.save(buf, "PNG")
    # plt.close(fig)
    # response = HttpResponse(buf.getvalue(), content_type='image/png')
    # return response

    # --display in html page--
    plt.savefig('static/usrlocation/images/g1.png')
    return render(request, "graph.html")
