import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the great-circle distance between two points 
    on the Earth in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def build_adjacency_matrix(locations, threshold_km=5000):
    """
    Builds a Gaussian Thresholded Adjacency Matrix from a list of location dictionaries.
    
    Args:
        locations (list of dicts): Must contain 'lat' and 'lon' keys.
        threshold_km (int): Maximum distance to consider an edge valid.
        
    Returns:
        numpy.ndarray: The weighted adjacency matrix.
    """
    n = len(locations)
    distances = np.zeros((n, n))
    
    # 1. Calculate all pairwise geographic distances
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = haversine(
                    locations[i]['lon'], locations[i]['lat'],
                    locations[j]['lon'], locations[j]['lat']
                )
    
    # 2. Calculate standard deviation of distances (sigma) for the Gaussian kernel
    sigma = np.std(distances[distances > 0])
    if sigma == 0: 
        sigma = 1 # Prevent division by zero
        
    adjacency_matrix = np.zeros((n, n))
    
    # 3. Apply the Gaussian Kernel and Threshold
    for i in range(n):
        for j in range(n):
            if distances[i, j] <= threshold_km:
                # Closer cities get weights closer to 1.0; distant cities drop toward 0.0
                adjacency_matrix[i, j] = np.exp(-(distances[i, j]**2) / (sigma**2))
            else:
                adjacency_matrix[i, j] = 0.0
                
    # 4. Fill diagonal with 1.0 (A region connects to itself perfectly)
    np.fill_diagonal(adjacency_matrix, 1.0) 
    
    return adjacency_matrix