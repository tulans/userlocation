import googlemaps
import matplotlib
import pandas as pd
import mpld3
import requests
import gmplot
from django.shortcuts import render
from django.template.response import TemplateResponse
from django.template import loader
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
import json
import time
# import io
import numpy as np
from shapely.geometry import MultiPoint
from geopy.distance import great_circle
# import seaborn as sns
# import matplotlib.pyplot as plt
from django.http import HttpResponse
# from matplotlib.backends.backend_agg import FigureCanvasAgg
from .forms import EmailForm

matplotlib.use('Agg')
plt = matplotlib.pyplot
kms_per_radian = 6371.0088

def get_centermost_point(cluster):
    centroid = (
        MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster,
                           key=lambda point: great_circle(point,
                                                          centroid).m)
    return tuple(centermost_point)

def call_argus(email):
    url = 'http://10.66.29.42:8087/argus/get/history'
    header = {'authorization': 'P0s6jZGIkwMrnS2',
              'accept': 'application/json',
              'content-type': 'application/json',
              'x-request-tracker': 'testIndc1'}
    req_payload = {"email": email}
    response = requests.post(url, data=json.dumps(req_payload),
                             headers=header, verify=False)
    return response


def get_dataset(file_name):
    with open(file_name) as f:
        r_json = json.load(f)
    return r_json

def get_emailid(request):
    if request.method == "POST":
        form = EmailForm(request.POST)
        if form.is_valid():
            try:
                email = form.data["email_id"]
                argus_res = call_argus(email)
                if argus_res.status_code != 200:
                    return TemplateResponse(
                        request,
                        "graph.html",
                        {"form": form, "error": "Error while calling Argus"},
                    )
                r_json = argus_res.json()
                if not r_json:
                    return

                # r_json = get_dataset('locations/userdata.json')
                data_json = pd.DataFrame.from_dict(r_json['data'], orient='columns')
                data = pd.DataFrame.from_dict(data_json, orient='columns')
                data['date'] = pd.to_datetime(data['time'], unit='ms')
                data['hour'] = data['date'].dt.hour

                # Print some stats about the data and also update the data dataframe with os_id
                print("Number of Unique OS types that user uses ", data['os'].unique())
                print("Unique Device Id count ", data['deviceId'].unique())
                print("Total data set for location that would be analyzed ", data['uuid'].count())
                data['os_id'] = data['os'].map({'Android': 0, 'ANDROID': 0, 'iOS': 1, 'IOS': 1})

                # Use DBSCAN Haversine metric algo to calculate the distance between various points on the graph and
                # cluster them together to identify the number of clusters.
                coords = data[['latitude', 'longitude']].to_numpy(float)
                epsilon = 1.5 / kms_per_radian  # this is the unit that has to be fed to DBSCAN for haversine calculations.
                db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                            metric='haversine').fit(np.radians(coords))
                cluster_labels = db.labels_
                num_clusters = len(set(cluster_labels))
                print(num_clusters)

                data['cluster_labels'] = cluster_labels
                data['cluster_id'] = cluster_labels

                cluster_count_dataframe = data.groupby('cluster_labels')['cluster_id'].count()

                start_time = time.time()
                message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
                print(message.format(len(data), num_clusters,
                                     100 * (1 - float(num_clusters) / len(data)),
                                     time.time() - start_time))
                print('Silhouette coefficient: {:0.03f}'.format(
                    metrics.silhouette_score(coords, cluster_labels)))

                # turn the clusters in to a pandas series, where each element is a cluster of points
                clusters = pd.Series(
                    [coords[cluster_labels == n] for n in range(num_clusters)])

                # Find the point in each cluster which is closest to its centroid
                centermost_points = clusters.map(get_centermost_point)
                print("Centermost points \n", centermost_points)
                # unzip the list of centermost points (lat, lon) tuples into separate lat and lon lists
                lats, lons = zip(*centermost_points)
                # from these lats/lons create a new df of one representative point for each cluster
                rep_points = pd.DataFrame({'longitude': lons, 'latitude': lats})
                print(rep_points.tail())
                # pull row from original data set where lat/lon match the lat/lon of each row of representative points
                # that way we get the full details from the original dataframe
                #
                data['latitude'] = data['latitude'].astype(float)
                data['longitude'] = data['longitude'].astype(float)
                rs = rep_points.merge(data, how='inner', on=['latitude', 'longitude'])
                rs['cluster_count'] = cluster_count_dataframe
                print(rs.head())

                # plot the final reduced set of coordinate points vs the original full set
                fig, ax = plt.subplots(figsize=[10, 10])
                rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='g', edgecolor='None', alpha=0.7, s=120)
                df_scatter = ax.scatter(data['longitude'], data['latitude'], c='k', alpha=0.9, s=3)
                ax.set_title(
                    'Actual Data Set Vs Filtered Data Set using Unsupervised Learning Technique (Clustered - DBSCAN)')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend([rs_scatter, df_scatter], ['Clustered', 'Non Clustered'], loc='upper right')
                plt.xticks(rotation=90)

                plt.savefig('static/usrlocation/images/g3.png')

                #fig 4:
                plt.figure(figsize=(10, 10))
                plt.scatter(rs['longitude'], rs['latitude'], c=rs['cluster_count'], label=rs['cluster_count'],
                            cmap='viridis')
                plt.colorbar()
                plt.title('Filtered User location')
                plt.xlabel('longitude')
                plt.ylabel('latitude')
                plt.xticks(rotation=90)
                # plt.legend()
                plt.savefig('static/usrlocation/images/g4.png')

                gmap = gmplot.GoogleMapPlotter(rs['latitude'].iloc[0],
                                               rs['longitude'].iloc[0], 5)
                gmap.apikey = 'AIzaSyBVWUMz35OVBSu8jQrkXGXpFu2z_R7fIJU'
                # gmap.plot(rs['latitude'], rs['longitude'], 'cornflowerblue', edge_width=1)
                gmap.scatter(rs['latitude'], rs['longitude'], 'k', marker=True)
                gmap.heatmap(rs['latitude'], rs['longitude'])
                gmap.draw(r'static/usrlocation/helper_templates/map.html')


                glocation = []
                # Google maps integration
                for index, row in rs.iterrows():
                    googlemps = googlemaps.Client(key='AIzaSyBVWUMz35OVBSu8jQrkXGXpFu2z_R7fIJU')
                    address = googlemps.reverse_geocode((row.latitude, row.longitude))
                    if address is not None:
                        gaddress_str = address[0]['formatted_address']
                        address_list = address[0]['address_components']
                        gpostal_code = ""
                        gcountry = ""
                        for li in address_list:
                            if "postal_code" in li['types']:
                                gpostal_code = li['long_name']
                            if "country" in li['types']:
                                gcountry = li['long_name']
                        glocation.append((gaddress_str, gpostal_code, gcountry))
                    else:
                        glocation.append("NaN")

                gloc_pd = pd.DataFrame(glocation, columns=('gaddress', 'gpostal_code', 'gcountry'))
                rs['gaddress'] = gloc_pd['gaddress']
                rs['gpostal_code'] = gloc_pd['gpostal_code']
                rs['gcountry'] = gloc_pd['gcountry']
                print("\n Top Postal Code visited by the user \n", rs.gpostal_code.unique())
                print("\n\n Top 10 Region/Area/Places that user visits by GOOGLE \n", rs[['gpostal_code']],
                      rs[['gaddress']])

                rs.to_json('locations/userdata_rs.json', orient='records')

                if email:
                    template_data = {
                        "form": form,
                        "email": email,
                    }
                    return TemplateResponse(request, "graph.html", template_data)
                else:
                    template_data = {
                        "form": form,
                        "error": "Please enter email ID",
                    }
                    return TemplateResponse(request, "graph.html", template_data)
            except Exception as e:
                print(e)
    else:
        form = EmailForm()

    return render(request, "graph.html", {"form": form})


def plot_graph(request):
    json = "locations/userdata_rs.json"
    rs = pd.read_json(json)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(rs['longitude'], rs['latitude'], c=rs['cluster_count'], label=rs['cluster_count'], cmap='viridis')
    plt.colorbar()
    plt.title('Filtered User location')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.xticks(rotation=90)
    # plt.legend()
    #plt.show()
    fig_html = mpld3.fig_to_html(fig)  # When we have local mpld3 libraries we will need to tweak this
    plt.savefig('static/usrlocation/images/g2.svg')
    return HttpResponse(fig_html)
    # print(fig_html)
    # return render(request, "graph.html", {"graph": fig_html, 'single_chart': single_chart})



from geopy.geocoders import Nominatim
def getDetailsByGeoPy(rs):
    location_node = []
    loc_pd = pd.DataFrame(location_node, columns=('city', 'postcode', 'country', 'address'))
    geolocator = Nominatim(user_agent='TestAppForDevices')

    for index, row in rs.iterrows():
        locat_str = str(row.latitude) + "," + str(row.longitude)
        location = geolocator.reverse(locat_str)
        address = location.raw['address']
        location_node.append((address.get('city', ''), address.get('postcode', ''), address.get('country', ''), location.address))


    rs['address'] = loc_pd['address']
    rs['city'] = loc_pd['city']
    rs['postcode'] = loc_pd['postcode']
    rs['country'] = loc_pd['country']

    print("\n Top Cities visited by the user \n", rs.city.unique())
    print("\n\n Top 10 Region/Area/Places that user visits by GEO PY \n", rs[['postcode']], rs[['city']],
          rs[['address']])