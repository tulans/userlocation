from . import views
from django.conf.urls import url

urlpatterns = [
    url(r"^plot_location_graph/?$", views.plot_graph),
    url(r"^index/?$", views.get_emailid),
    url(r"^gmap_details/?$", views.gmap_details),
]
