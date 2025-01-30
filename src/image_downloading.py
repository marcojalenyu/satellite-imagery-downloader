import cv2
import requests
import numpy as np
import threading
import os


def download_tile(url, headers, channels, path):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        if channels == 3:
            tile = cv2.imdecode(arr, 1)
        else:
            tile = cv2.imdecode(arr, -1)
        
        if tile is None:
            raise ValueError("Failed to decode image")
        
        # Save the image to disk
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        cv2.imwrite(path, tile)
        
        return tile
    
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError as e:
        print(f"Decoding failed: {e}")
        return None


# Mercator projection 
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
def project_with_scale(lat, lon, scale):
    siny = np.sin(lat * np.pi / 180)
    siny = min(max(siny, -0.9999), 0.9999)
    x = scale * (0.5 + lon / 360)
    y = scale * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))
    return x, y


def download_image(lat1: float, lon1: float, lat2: float, lon2: float,
    zoom: int, url: str, headers: dict, tile_size: int = 256, channels: int = 3, directory: str = None) -> np.ndarray:
    """
    Downloads a map region. Returns an image stored as a `numpy.ndarray` in BGR or BGRA, depending on the number
    of `channels`.

    Parameters
    ----------
    `(lat1, lon1)` - Coordinates (decimal degrees) of the top-left corner of a rectangular area

    `(lat2, lon2)` - Coordinates (decimal degrees) of the bottom-right corner of a rectangular area

    `zoom` - Zoom level

    `url` - Tile URL with {x}, {y} and {z} in place of its coordinate and zoom values

    `headers` - Dictionary of HTTP headers

    `tile_size` - Tile size in pixels

    `channels` - Number of channels in the output image. Also affects how the tiles are converted into numpy arrays.
    """

    scale = 1 << zoom

    # Find the pixel coordinates and tile coordinates of the corners
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    tl_tile_x = int(tl_proj_x)
    tl_tile_y = int(tl_proj_y)
    br_tile_x = int(br_proj_x)
    br_tile_y = int(br_proj_y)

    img_w = abs(tl_pixel_x - br_pixel_x)
    img_h = br_pixel_y - tl_pixel_y
    img = np.zeros((img_h, img_w, channels), np.uint8)

    # Print the pixel coordinates of the corners
    print(tl_pixel_x, tl_pixel_y, br_pixel_x, br_pixel_y)
    # Print the tile coordinates of the corners
    print(tl_tile_x, tl_tile_y, br_tile_x, br_tile_y)
    # Print longitude and latitude of the corners
    print(lat1, lon1, lat2, lon2)

    def build_row(tile_y, tile_size):
        for tile_x in range(tl_tile_x, br_tile_x + 1):
            # Calculate the latitude and longitude for the current tile
            tile_lat = lat1 + (lat2 - lat1) * (tile_y - tl_tile_y) / (br_tile_y - tl_tile_y)
            tile_lon = lon1 + (lon2 - lon1) * (tile_x - tl_tile_x) / (br_tile_x - tl_tile_x)
                
            # Create the path with latitude and longitude in the filename
            path = os.path.join(directory, f"_{tile_lat:.6f}_{tile_lon:.6f}.png")
            
            tile = download_tile(url.format(x=tile_lat, y=tile_lon, z=zoom), headers, channels, path)

            if tile is not None:
                # Find the pixel coordinates of the new tile relative to the image
                tl_rel_x = tile_x * tile_size - tl_pixel_x
                tl_rel_y = tile_y * tile_size - tl_pixel_y
                br_rel_x = tl_rel_x + tile_size
                br_rel_y = tl_rel_y + tile_size
                
                # Define where the tile will be placed on the image
                img_x_l = max(0, tl_rel_x)
                img_x_r = min(img_w, br_rel_x)
                img_y_l = max(0, tl_rel_y)
                img_y_r = min(img_h, br_rel_y)

                # Define how border tiles will be cropped
                cr_x_l = max(0, -tl_rel_x)
                cr_x_r = tile_size + min(0, img_w - br_rel_x)
                cr_y_l = max(0, -tl_rel_y)
                cr_y_r = tile_size + min(0, img_h - br_rel_y)

                img[img_y_l:img_y_r, img_x_l:img_x_r] = tile[cr_y_l:cr_y_r, cr_x_l:cr_x_r]


    threads = []
    for tile_y in range(tl_tile_y, br_tile_y + 1):
        thread = threading.Thread(target=build_row, args=[tile_y, tile_size])
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    return img


def image_size(lat1: float, lon1: float, lat2: float,
    lon2: float, zoom: int, tile_size: int = 256):
    """ Calculates the size of an image without downloading it. Returns the width and height in pixels as a tuple. """

    scale = 1 << zoom
    tl_proj_x, tl_proj_y = project_with_scale(lat1, lon1, scale)
    br_proj_x, br_proj_y = project_with_scale(lat2, lon2, scale)

    tl_pixel_x = int(tl_proj_x * tile_size)
    tl_pixel_y = int(tl_proj_y * tile_size)
    br_pixel_x = int(br_proj_x * tile_size)
    br_pixel_y = int(br_proj_y * tile_size)

    print(tl_pixel_x, tl_pixel_y, br_pixel_x, br_pixel_y)

    return abs(tl_pixel_x - br_pixel_x), br_pixel_y - tl_pixel_y
