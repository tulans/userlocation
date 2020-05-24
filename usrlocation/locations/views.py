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

loc_data = None

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
    res = requests.post(url, data=json.dumps(req_payload),
                             headers=header, verify=False)
    if res.status_code != 200:
        return None
    response = res.json()
    return response

def create_dataframe(r_json):
    data_json = pd.DataFrame.from_dict(r_json['data'], orient='columns')
    data = pd.DataFrame.from_dict(data_json, orient='columns')
    data['date'] = pd.to_datetime(data['time'], unit='ms')
    data['hour'] = data['date'].dt.hour

    coords = data[['latitude', 'longitude']].to_numpy(float)
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    print(num_clusters)

    # start_time = time.time()
    # message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
    # print(message.format(len(data), num_clusters,
    #                      100 * (1 - float(num_clusters) / len(data)),
    #                      time.time() - start_time))
    # print('Silhouette coefficient: {:0.03f}'.format(
    #     metrics.silhouette_score(coords, cluster_labels)))

    clusters = pd.Series(
        [coords[cluster_labels == n] for n in range(num_clusters)])

    centermost_points = clusters.map(get_centermost_point)
    print("Centermost points \n", centermost_points)
    lats, lons = zip(*centermost_points)
    rep_points = pd.DataFrame({'longitude': lons, 'latitude': lats})
    # print(rep_points.tail())
    return rep_points

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
                # r_json = call_argus(email)
                # if not r_json:
                #     return TemplateResponse(
                #         request,
                #         "graph.html",
                #         {"form": form, "error": "Error while calling Argus"},
                #     )

                r_json = get_dataset('locations/userdata.json')
                data_json = pd.DataFrame.from_dict(r_json['data'], orient='columns')
                data = pd.DataFrame.from_dict(data_json, orient='columns')
                data['date'] = pd.to_datetime(data['time'], unit='ms')
                data['hour'] = data['date'].dt.hour

                coords = data[['latitude', 'longitude']].to_numpy(float)
                epsilon = 1.5 / kms_per_radian
                db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                            metric='haversine').fit(np.radians(coords))
                cluster_labels = db.labels_
                num_clusters = len(set(cluster_labels))
                data['cluster_labels'] = cluster_labels
                data['cluster_id'] = cluster_labels
                cluster_count_dataframe = data.groupby('cluster_labels')[
                    'cluster_id'].count()
                print(num_clusters)

                # start_time = time.time()
                # message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
                # print(message.format(len(data), num_clusters,
                #                      100 * (1 - float(num_clusters) / len(data)),
                #                      time.time() - start_time))
                # print('Silhouette coefficient: {:0.03f}'.format(
                #     metrics.silhouette_score(coords, cluster_labels)))

                clusters = pd.Series(
                    [coords[cluster_labels == n] for n in range(num_clusters)])

                centermost_points = clusters.map(get_centermost_point)
                print("Centermost points \n", centermost_points)
                lats, lons = zip(*centermost_points)
                rep_points = pd.DataFrame({'longitude': lons, 'latitude': lats})
                # print(rep_points.tail())

                data['latitude'] = data['latitude'].astype(float)
                data['longitude'] = data['longitude'].astype(float)
                rs = rep_points.merge(data, how='inner', on=['latitude', 'longitude'])
                rs['cluster_count'] = cluster_count_dataframe

                out_json1 = data.to_json(orient='records')
                with open('locations/rs1.json', 'w') as outfile1:
                    json.dump(out_json1, outfile1)
                out_json = rs.to_json(orient='records')
                with open('locations/rs.json', 'w') as outfile:
                    json.dump(out_json, outfile)

                # rs = rep_points.apply(lambda row: data[(data['latitude'] == row['latitude']) and (data['longitude'] == row['longitude'])].iloc[0], axis=1)
                # rs = rep_points.copy()
                # print(rs.head())
                # plot the final reduced set of coordinate points vs the original full set


                #fig1
                fig, ax = plt.subplots(figsize=[10, 10])
                rs_scatter = ax.scatter(rs['longitude'], rs['latitude'], c='g',
                                        edgecolor='None', alpha=0.7, s=120)
                df_scatter = ax.scatter(data['longitude'], data['latitude'], c='k',
                                        alpha=0.9, s=3)
                ax.set_title('Actual Data Set Vs Filtered Data Set using Unsupervised Learning Technique (Clustered - DBSCAN)')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend([rs_scatter, df_scatter], ['Clustered', 'Non Clustered'],
                          loc='upper right')
                plt.xticks(rotation=90)

                plt.savefig('static/usrlocation/images/g3.png')

                #fig2

                plt.figure(figsize=(10, 10))
                plt.scatter(rs['longitude'], rs['latitude'], c=rs['cluster_count'],
                            label=rs['cluster_count'], cmap='viridis')
                plt.colorbar()
                plt.title('Filtered User location')
                plt.xlabel('longitude')
                plt.ylabel('latitude')
                plt.xticks(rotation=90)
                plt.savefig('static/usrlocation/images/g4.png')


                gmap = gmplot.GoogleMapPlotter(rs['latitude'].iloc[0],
                                               rs['longitude'].iloc[0], 6)
                gmap.apikey = 'AIzaSyBVWUMz35OVBSu8jQrkXGXpFu2z_R7fIJU'
                # gmap.plot(rs['latitude'], rs['longitude'], 'cornflowerblue', edge_width=1)
                # gmap.scatter(rs['latitude'], rs['longitude'], 'k', marker=True)
                gmap.heatmap(rs['latitude'], rs['longitude'])
                gmap.draw(r'static/usrlocation/helper_templates/map.html')

                if email:
                    template_data = {
                        "form": form,
                        "email": email
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
    # csv_file = "locations/CountriesExerciseCluster.csv"
    # data = pd.read_csv(csv_file)
    r_json = json.loads(get_dataset('locations/rs.json'))
    data_json = pd.DataFrame.from_dict(r_json, orient='columns')
    rs = pd.DataFrame.from_dict(data_json, orient='columns')
    print(rs)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(rs['longitude'], rs['latitude'], c=rs['cluster_count'],
                label=rs['cluster_count'], cmap='viridis')
    plt.colorbar()
    plt.title('Filtered User location')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.xticks(rotation=90)
    # plt.savefig('static/usrlocation/images/g4.png')

    # plt.scatter(data['Longitude'], data['Latitude'], c=data['Cluster'], cmap='rainbow')

    fig_html = mpld3.fig_to_html(fig)  # When we have local mpld3 libraries we will need to tweak this
    # plt.savefig('static/usrlocation/images/g2.svg')
    return HttpResponse(fig_html)

def gmap_details(request):
    import googlemaps
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    form = EmailForm(request.POST)
    location_node = []
    glocation = []
    # r_json = json.loads(get_dataset('locations/rs1.json'))
    r_json = json.loads(get_dataset('locations/rs.json'))
    data_json = pd.DataFrame.from_dict(r_json, orient='columns')
    rs = pd.DataFrame.from_dict(data_json, orient='columns')
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

    geolocator = Nominatim(user_agent='TestAppForDevices')
    for index, row in rs.iterrows():
        locat_str = str(row.latitude) + "," + str(row.longitude)
        try:
            location = geolocator.reverse(locat_str, timeout=10)
        except GeocoderTimedOut as e:
            print("Error: geocode failed on input %s with message %s; Retrying" % (
                locat_str, e))
            location = geolocator.reverse(locat_str)

        address = location.raw['address']
        location_node.append((address.get('city', ''), address.get('postcode', ''),
                              address.get('country', ''), location.address))
    loc_pd = pd.DataFrame(location_node,
                          columns=('city', 'postcode', 'country', 'address'))
    gloc_pd = pd.DataFrame(glocation, columns=('gaddress', 'gpostal_code', 'gcountry'))
    rs['address'] = loc_pd['address']
    rs['city'] = loc_pd['city']
    rs['postcode'] = loc_pd['postcode']
    rs['country'] = loc_pd['country']
    rs['gaddress'] = gloc_pd['gaddress']
    rs['gpostal_code'] = gloc_pd['gpostal_code']
    rs['gcountry'] = gloc_pd['gcountry']
    print("\n Top Cities visited by the user \n", rs.city.unique())
    print("\n\n Top 10 Region/Area/Places that user visits by GEO PY \n",
          rs[['postcode']], rs[['city']], rs[['address']])
    print("\n Top Postal Code visited by the user \n", rs.gpostal_code.unique())
    print("\n\n Top 10 Region/Area/Places that user visits by GOOGLE \n",
          rs[['gpostal_code']], rs[['gaddress']])

    rs['date'] = pd.to_datetime(rs['time'], unit='ms')
    rs['hour'] = rs['date'].dt.hour

    r_json1 = json.loads(get_dataset('locations/rs1.json'))
    data_json1 = pd.DataFrame.from_dict(r_json1, orient='columns')
    rs1 = pd.DataFrame.from_dict(data_json1, orient='columns')

    coords = rs1[['latitude', 'longitude']].to_numpy(float)
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree',
                metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    print(num_clusters)

    start_time = time.time()
    message = 'Clustered {:,} points down to {:,} clusters, for {:.1f}% compression in {:,.2f} seconds'
    msg = message.format(len(rs1), num_clusters,
                         100 * (1 - float(num_clusters) / len(rs1)),
                         time.time() - start_time)
    silcoeff = 'Silhouette coefficient: {:0.03f}'.format(
        metrics.silhouette_score(coords, cluster_labels))

    data_json = json.loads(rs.to_json(orient='records'))
    data = {
        "uniqueOS": rs1['os'].unique(),
        "uniqueDID": rs1['deviceId'].unique(),
        "totloc": rs1['uuid'].count(),
        "message": msg,
        "silcoeff": silcoeff,
        "topcities": rs.city.unique(),
        "placedetails": data_json,
        "toppc": rs.gpostal_code.unique(),
    }
    return render(request, "graph.html", {"form": form, "gmdata": data})
