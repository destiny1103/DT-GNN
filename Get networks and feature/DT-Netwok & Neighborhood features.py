import igraph as ig
import pandas as pd
import numpy as np
import shapely.geometry
from shapely.ops import unary_union
from scipy import sparse
from scipy.spatial import Delaunay, Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from geopy.distance import geodesic as GD
import itertools
from itertools import chain
import math
from tqdm import tqdm
import pyproj

# Load data
df = pd.read_csv('.\\data\\NewYork\\Feature1&2&3(count).csv')

# Getting the network node dataset from the data
AIS_MMSI = df['MMSI'].unique()
point_list = []
for MMSI in AIS_MMSI:
    one_Cargo = df[(df.MMSI == MMSI)].reset_index().drop(columns=['index'])
    for index, row in one_Cargo.iterrows():
        point_list.append([row['LAT'], row['LON'], row['MMSI']])
points = np.array(point_list)

# Introduction to function crateDTNetwork
# 1) Compute Delaunay triangulation of point sets.
# 2) Network construction based on Delaunay triangulation results.
# 3) Function input: Scattered latitude and longitude arrays
# 4) Function output: igraph network, Delaunay triangular mesh indexing
def crateDTNetwork(points):
    g = ig.Graph(len(points))  # Defining the Network
    # Adding nodes and node attributes to the network
    g.vs['y'] = points[:, 0]
    g.vs['x'] = points[:, 1]
    layout = g.layout_auto()  # Wrapping the latitude and longitude attributes of a node

    # Delaunay triangulation
    delaunay = Delaunay(layout.coords)

    # Adding edges to a network (based on Delaunay triangulation)
    edges = set()
    for tri_index in delaunay.simplices:
        edges.update(itertools.combinations(tri_index, 2))
    g.add_edges(list(edges))
    g.simplify()
    return g, delaunay

# Constructing the ship trajectory network
g, delaunay = crateDTNetwork(points)

# Delete Long Edge
lenght = 1  # Sets the interruption threshold. Unit: km (default: 1 km)
i, n = 0, 0
b = g.get_edge_dataframe()
delete_edge = []  # Gets the deleted edges (in the form of node index OD) for removing redundant triangles from the original Delaunay triangulation result.
for i in range(len(b)):
    if GD([g.vs[b['source'][i]]['y'], g.vs[b['source'][i]]['x']],
          [[g.vs[b['target'][i]]['y'], g.vs[b['target'][i]]['x']]]) > lenght:
        g.delete_edges(i - n)
        delete_edge.append([b['source'][i], b['target'][i]])
        n = n + 1


############ Output graph sparse matrix file for model training (Graph_sparse_matrix.npz) ############

# Add weight information to the network edges, with the edge weights being the reciprocal of the distance between the trajectory points. Units
def eg_wighted(point1, point2):
    lat1, lon1 = g.vs[point1]['y'], g.vs[point1]['x']
    lat2, lon2 = g.vs[point2]['y'], g.vs[point2]['x']
    edgewighted = GD([lat1, lon1], [lat2, lon2])
    return edgewighted
edge_weight = []
for edge in g.es:
    egwighted = eg_wighted(edge.source, edge.target)
    edge_weight.append(1 / egwighted.m)
g.es['weight'] = edge_weight

# Creating sparse matrices in CSR (Compressed Sparse Row) format
n = g.vcount()
indptr = [0]
indices = []
data = []
for i in range(n):
    neighbors = g.neighbors(i, mode='out')
    edges = g.es[g.incident(i, mode='out')]
    weights = [edge['weight'] for edge in edges]
    indptr.append(indptr[-1] + len(neighbors))
    indices.extend(neighbors)
    data.extend(weights)

# Convert CSR format to NPZ format and output. NPZ files are compressed NumPy array files.
sparse_matrix = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
# sparse.save_npz(".\\data\\NewYork\\Graph_sparse_matrix.npz", sparse_matrix)

########################################################################################################################
# The following code is used to compute the spatial neighborhood features of a node.
# The Euclidean spatial features contain n-order neighborhoods and Voronoi correlation features.
# The steps are as follows:
# 1) Generation of outer polygons for scatter networks.
# This step is performed to eliminate disturbances in the computation of features
# (uneven number and area of triangles in the exterior of the nth-order neighborhood, disturbances in the area by ultra-distant points in Voronoi).
# 2) Calculate n-order neighborhood features.
# Include the number of triangles in the neighborhood, the area, the mean of the area, the standard deviation of the area, and the coefficient of variation of the area
# 3) Calculate Voronoi neighborhood features.

##################### step 1) Generation of outer polygons for scatter networks #####################
tris = delaunay.simplices
tris = tris.tolist()
# Delete a subset of the corresponding triangle index array in tris (delaunay.simplices) by the deleted long edge
d_edge_index = []
for d_edge in delete_edge:
    for i in range(len(tris) - 1):
        if set(d_edge).issubset(set(tris[i])):
            # print(d_edge, tris[i], i)
            d_edge_index.append(i)
for i in sorted(set(d_edge_index), reverse=True):
    del tris[i]
# print(len(tris))

# Construct scatter polygon using triangle mesh (with long edges removed)
# step 1.Triangle merge
# step 2.Polygon object merge
All_polygon = []
for i in range(len(tris) - 1):
    tri = tris[i]
    y1, x1 = g.vs[tri[0]]['x'], g.vs[tri[0]]['y']
    y2, x2 = g.vs[tri[1]]['x'], g.vs[tri[1]]['y']
    y3, x3 = g.vs[tri[2]]['x'], g.vs[tri[2]]['y']
    polygon = shapely.Polygon([[y1, x1], [y2, x2], [y3, x3]])
    All_polygon.append(polygon)
multi_polygon = shapely.unary_union(All_polygon)
if type(multi_polygon) == Polygon:
    print("polygon is a Polygon object")
    Connection_polygon = multi_polygon
else:
    print("polygon is a multipolygon object")
    Connection_polygon = shapely.geometry.Polygon()
    for i in range(len(multi_polygon.geoms) - 1):
        closest_points = shapely.ops.nearest_points(multi_polygon.geoms[i], multi_polygon.geoms[i + 1])
        line = shapely.geometry.LineString(closest_points)
        if line.length == 0:
            buffer_zone = line.buffer(10 / (111000 * np.cos(np.deg2rad(40.65))))
            Connection_polygon = shapely.unary_union(
                [Connection_polygon, multi_polygon.geoms[i], buffer_zone, multi_polygon.geoms[i + 1]])
        else:
            buffer_zone = line.buffer(10 / (111000 * np.cos(np.deg2rad(40.65))))
            Connection_polygon = shapely.unary_union(
                [Connection_polygon, multi_polygon.geoms[i], buffer_zone, multi_polygon.geoms[i + 1]])

# First layer of buffered boundaries to supplement spatial information at boundary points
Connection_polygon1 = Connection_polygon.buffer(20 / (111000 * np.cos(np.deg2rad(40.65))))
# Second buffer boundary to constrains the spatial characteristics of the first buffer points
Connection_polygon2 = Connection_polygon.buffer(40 / (111000 * np.cos(np.deg2rad(40.65))))

# line encryption function
def get_fill_points(sp, ep, d, r, c):
    lng_diff = ep[1] - sp[1]
    lat_diff = ep[0] - sp[0]
    n = math.floor(d / r)
    a = lng_diff / n
    b = lat_diff / n
    points = []
    for i in range(1, n):
        lng = sp[1] + a * i
        lat = sp[0] + b * i
        points.append([lat, lng])
    if c:
        points.insert(0, sp.tolist())
        # points.append(ep)
    return points


# Contour Edge Encryption Function (Inputs: encrypted point set, encryption distance thresholds)
def insert_points(buffered_boundary_polygon_coords, buffer_distance):
    all_insert_points = []
    for i in range(len(buffered_boundary_polygon_coords) - 1):
        start_point = buffered_boundary_polygon_coords[i]
        end_point = buffered_boundary_polygon_coords[i + 1]
        # distance = GD([start_point[0], start_point[1]], [end_point[0], end_point[1]]).m
        distance = GD([start_point[1], start_point[0]], [end_point[1], end_point[0]]).m
        contain_both = True  # 包含两端
        if distance > 1.5 * buffer_distance:
            insert_points = get_fill_points(start_point, end_point, distance, buffer_distance, contain_both)
            all_insert_points = all_insert_points + insert_points
        else:
            all_insert_points = all_insert_points + [start_point.tolist()]
    return all_insert_points


insert_Connection_polygon1 = insert_points(np.array(Connection_polygon1.exterior.coords), 20)  # first-order buffer profile encryption
insert_Connection_polygon2 = insert_points(np.array(Connection_polygon2.exterior.coords), 20)  # second-order buffer profile encryption

AIS_points = np.stack(points[:, :2]).tolist()
insert_Connection_polygon1_point = [[b, a] for a, b in insert_Connection_polygon1]
insert_Connection_polygon2_point = [[b, a] for a, b in insert_Connection_polygon2]
new_points = AIS_points + insert_Connection_polygon1_point + insert_Connection_polygon2_point


# # Visual triangular mesh and two-layer buffer zone
# x1, y1 = [i[0] for i in insert_Connection_polygon1], [i[1] for i in insert_Connection_polygon1]
# x2, y2 = [i[0] for i in insert_Connection_polygon2], [i[1] for i in insert_Connection_polygon2]
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# # ax.add_patch(plt.Polygon(list(Connection_polygon1.exterior.coords), edgecolor='red', alpha=0.5))
# # plt.scatter(x1, y1, c='blue', alpha=0.5, marker='o')
# # ax.add_patch(plt.Polygon(list(Connection_polygon2.exterior.coords), edgecolor='green', alpha=0.5))
# # plt.scatter(x2, y2, c='blue', alpha=0.5, marker='o')
# plt.triplot(points[:, 1], points[:, 0], tris, c='black', linewidth=0.1)
# plt.plot(points[:, 1], points[:, 0], '*', c='lightcoral', markersize=1)
# # NewYork
# # ax.set_xlim(-74.4, -73.8)
# # ax.set_ylim(40.4, 40.8)
# plt.show()
# sys.exit()


##################### step 2) Calculate n-order neighborhood features #####################
# Find all triangles containing a node from the Delaunay triangulation result (array)
def find_element(array, element):
    arr = []
    for i, lst in enumerate(array):
        if element in lst:
            arr.append(i)
            # return i, lst.index(element)
    return arr


# Helen's formula for finding the area of a triangle (enter latitude and longitude coordinates)
def tri_Area(one_tri, g):
    y1, x1 = g.vs[one_tri[0]]['y'], g.vs[one_tri[0]]['x']
    y2, x2 = g.vs[one_tri[1]]['y'], g.vs[one_tri[1]]['x']
    y3, x3 = g.vs[one_tri[2]]['y'], g.vs[one_tri[2]]['x']
    a = GD([y1, x1], [y2, x2]).km
    b = GD([y1, x1], [y3, x3]).km
    c = GD([y2, x2], [y3, x3]).km
    s = (a + b + c) / 2
    area = pow((s * (s - a) * (s - b) * (s - c)), 0.5)
    return area.real


# Calculation of n-order neighborhood properties
def neighborhood_tri_TotalandArea(All_tris, tris, g):
    tris_Area = []
    tris_points = []
    for tri in All_tris:
        one_tri_Area = tri_Area(tris[tri], g)
        tris_Area.append(one_tri_Area)
        tris_points.append(tris[tri])
    if len(All_tris) != 0:
        tris_Area_avg = np.mean(tris_Area)
        tris_Area_std = np.std(tris_Area)
        tris_Area_CV = np.std(tris_Area) / np.mean(tris_Area)
    else:
        tris_Area_avg, tris_Area_std, tris_Area_CV = 0, 0, 0
    return len(All_tris), tris_points, tris_Area_avg, tris_Area_std, tris_Area_CV


# n+1 order neighborhood triangle index value acquisition
def get_Seed_Point(Seed_Point, delete_tris_index, delete_points_index, tris):
    now_tris_index = set(chain.from_iterable(map(lambda i: find_element(tris, i), Seed_Point)))  # Seed point proximity triangle index
    now_tris_index = [i for i in now_tris_index if i not in delete_tris_index]  # Eliminate the last order neighborhood triangle index
    next_points_index = set(chain.from_iterable(map(lambda i: tris[i], now_tris_index)))  # Proximity triangle outer seed point acquisition
    # next_points_index = [i for i in next_points_index if i not in delete_points_index]
    # next_points_index = [i for i in next_points_index if i not in Seed_Point]
    next_points_index = list(set(next_points_index) - set(delete_points_index) - set(Seed_Point))
    return now_tris_index, next_points_index

new_points = np.array(new_points)
new_g, new_delaunay = crateDTNetwork(new_points)
new_tris = new_delaunay.simplices.tolist()
# Delete long edge (new network containing outer buffer)
lenght = 1
i, n = 0, 0
b = new_g.get_edge_dataframe()
delete_edge = []
for i in range(len(b)):
    if GD([new_g.vs[b['source'][i]]['y'], new_g.vs[b['source'][i]]['x']],
          [[new_g.vs[b['target'][i]]['y'], new_g.vs[b['target'][i]]['x']]]) > lenght:
        new_g.delete_edges(i - n)
        delete_edge.append([b['source'][i], b['target'][i]])
        n = n + 1

d_edge_index = []
for d_edge in delete_edge:
    for i in range(len(new_tris) - 1):
        if set(d_edge).issubset(set(new_tris[i])):
            # print(d_edge, tris[i], i)
            d_edge_index.append(i)
for i in sorted(set(d_edge_index), reverse=True):
    del new_tris[i]

# Visualize new networks & old points
# plt.triplot(new_points[:, 1], new_points[:, 0], new_tris, c='black', linewidth=0.1)
# plt.plot(new_points[:len(points), 1], new_points[:len(points), 0], '*', c='red', markersize=1)
# plt.show()

# Calculate the n-order neighborhood features of the node
count = 0
node_neighborhood_Features = []
for Origin in tqdm(range(len(g.vs))):  # Origin (Aim point)
    neighborhood_tri_node_index_1 = find_element(new_tris, Origin)
    Seed_Point2 = set(chain.from_iterable(map(lambda i: new_tris[i], neighborhood_tri_node_index_1)))

    try:
        Seed_Point2.remove(Origin)
    except:
        pass

    neighborhood_tri_node_index_2, Seed_Point3 = get_Seed_Point(Seed_Point2, neighborhood_tri_node_index_1,
                                                                [Origin], new_tris)  # Get 2-order neighborhood triangle index (the same as below)
    neighborhood_tri_node_index_3, Seed_Point4 = get_Seed_Point(Seed_Point3, neighborhood_tri_node_index_2,
                                                                Seed_Point2, new_tris)
    # neighborhood_tri_node_index_4, Seed_Point5 = get_Seed_Point(Seed_Point4, neighborhood_tri_node_index_3, Seed_Point3)

    # features calculation
    tri_Total1, tris_points1, tris_Area_avg1, tris_Area_std1, tris_Area_CV1 = neighborhood_tri_TotalandArea(
        neighborhood_tri_node_index_1, new_tris, new_g)
    tri_Total2, tris_points2, tris_Area_avg2, tris_Area_std2, tris_Area_CV2 = neighborhood_tri_TotalandArea(
        neighborhood_tri_node_index_2, new_tris, new_g)
    tri_Total3, tris_points3, tris_Area_avg3, tris_Area_std3, tris_Area_CV3 = neighborhood_tri_TotalandArea(
        neighborhood_tri_node_index_3, new_tris, new_g)
    # tri_Total_4, tri_Area_4, tris_points4 = neighborhood_tri_TotalandArea(neighborhood_tri_node_index_4)

    node_neighborhood_Features.append(
        [tri_Total1, tris_Area_avg1, tris_Area_std1, tris_Area_CV1, tri_Total2, tris_Area_avg2, tris_Area_std2,
         tris_Area_CV2, tri_Total3, tris_Area_avg3, tris_Area_std3, tris_Area_CV3])
    # break

# # Visualize the k-order neighborhood structure of a aim point
# # tris_points1 first-order neighborhood triangular mesh / tris_points2 Second-order neighborhood triangular networks / tris_points3 Third -order neighborhood triangular networks
# plt.triplot(new_points[:, 1], new_points[:, 0], tris, c='black', linewidth=0.05)
# plt.plot(new_points[:, 1], new_points[:, 0], '.', c='blue', markersize=0.1)
# plt.triplot(new_points[:, 1], new_points[:, 0], tris_points1, c='green', linewidth=1)
# plt.triplot(new_points[:, 1], new_points[:, 0], tris_points2, c='red', linewidth=1)
# plt.triplot(new_points[:, 1], new_points[:, 0], tris_points3, c='green', linewidth=1)
# # plt.triplot(points[:, 1], points[:, 0], tris_points4, c='red', linewidth=0.3)
# plt.axis('equal')
# plt.show()

node_neighborhood_Features = pd.DataFrame(node_neighborhood_Features)
node_neighborhood_Features.to_csv(".\\data\\NewYork\\feature3(k-order).csv", index=False, header=False)

##################### step 3) Calculate Voronoi neighborhood features #####################
project = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
xy = project.transform(new_points[:, 0], new_points[:, 1])
print(type(xy))
vor = Voronoi(np.array(xy).T)
# vor = Voronoi(new_points)
# fig, ax = plt.subplots()
# ax.triplot(new_points[:, 0], new_points[:, 1], new_tris, c='black', linewidth=0.1)
# ax.plot(new_points[:len(points), 0], new_points[:len(points), 1], '*', c='red', markersize=1)
# voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_widths=0.05)
# plt.show()

Voronoi_features = []
for i in tqdm(range(len(points))):
    vertices_idx = vor.regions[vor.point_region[i]]
    vertices = vor.vertices[vertices_idx]
    # Calculate the area (vector cross product) and the number of sides of an exterior polygon.
    area = 0.5 * np.abs(
        np.dot(vertices[:, 0], np.roll(vertices[:, 1], 1)) - np.dot(vertices[:, 1], np.roll(vertices[:, 0], 1)))
    area = min(area, 10000)
    num_edges = len(vertices)
    perimeter = 0
    for i in range(len(vertices)):
        edge = vertices[(i + 1) % len(vertices)] - vertices[i]
        edge_length = np.sqrt(np.sum(edge ** 2))
        perimeter += edge_length
    perimeter = min(perimeter, 2000)
    Voronoi_features.append([area, num_edges, perimeter])

Voronoi_features = pd.DataFrame(Voronoi_features)
Voronoi_features.to_csv(".\\data\\NewYork\\feature3(voronoi).csv", index=False,header=False)