import json
from typing import Dict, Any
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon
from pandas import DataFrame
from pathlib import Path
from typing import List
import consts as C


def make_crop(x, y, color, zoom, *args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    # Define the size of the crop region around the TFL
    crop_width = C.DEFAULT_CROPS_W
    crop_height = C.DEFAULT_CROPS_H
    y0_offset = 0
    y1_offset = 0
    # Adjust y0 offset and y1 offset based on color
    if color == 'r':
        y0_offset = 1 / 3
        y1_offset = 2 / 3
    elif color == 'g':
        y0_offset = 2 / 3
        y1_offset = 1 / 3

    # Calculate the default cropping region around the TFL
    x0 = int(x - crop_width // 2)
    x1 = int(x + crop_width // 2)
    y0 = int(y - (crop_height * y0_offset))
    y1 = int(y + (crop_height * y1_offset))

    return x0, x1, y0, y1, 'crop_data'


def check_crop(image_json_path, x0, x1, y0, y1):
    image_json_path = C.PART_IMAGE_SET / C.IMAGES_1 / image_json_path

    image_json = json.load(Path(image_json_path).open())
    traffic_light_polygons: List[C.POLYGON_OBJECT] = [image_object for image_object in image_json['objects']
                                                    if image_object['label'] in C.TFL_LABEL]
    is_true, ignore = False, False
    cropped_polygon = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    count_traffic_lights = 0
    for tl_polygon in traffic_light_polygons:
        if cropped_polygon.contains(Polygon(tl_polygon['polygon'])):
            count_traffic_lights += 1
            if count_traffic_lights >= 2:
                ignore = True
                break
            is_true = True

    return is_true, ignore


def save_for_part_2(crops_df: DataFrame):
    """
    *** No need to touch this. ***
    Saves the result DataFrame containing the crops data in the relevant folder under the relevant name for part 2.
    """
    if not C.ATTENTION_PATH.exists():
        C.ATTENTION_PATH.mkdir()
    crops_sorted: DataFrame = crops_df
    crops_sorted.to_csv(C.ATTENTION_PATH / C.CROP_CSV_NAME, index=False)


def create_crops(df: DataFrame, IGNOR=None) -> DataFrame:
    # creates a folder for you to save the crops in, recommended not must
    if not C.CROP_DIR.exists():
        C.CROP_DIR.mkdir()

    result_df = pd.DataFrame(columns=C.CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {C.SEQ: '', C.IS_TRUE: '', IGNOR: '', C.CROP_PATH: '', C.JSON_PATH: '', C.X0: '', C.X1: '', C.Y0: '', C.Y1: '',
                                  C.COL: ''}
    for index, row in df.iterrows():
        # Save sequence TFL in each image
        result_template[C.SEQ] = row[C.SEQ]
        # Save color TFL in each image
        result_template[C.COL] = row[C.COL]

        # Extract image_path
        image_path = row[C.CROP_PATH]

        # Extrac corp rect from
        x0, x1, y0, y1, crop = make_crop(row[C.X], row[C.Y], row[C.COL], row[C.ZOOM])
        result_template[C.X0], result_template[C.X1], result_template[C.Y0], result_template[C.Y1] = x0, x1, y0, y1

        # crop.save(CROP_DIR / crop_path)

        # Save json path (Anton need to pass from part 1)
        image_json_path = image_path.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        result_template[C.JSON_PATH] = image_json_path

        # Check crop rectangle if it TFL or not, ignore if it parts of TFL, double TFL,
        result_template[C.IS_TRUE], result_template[C.IS_IGNORE] = check_crop(image_json_path, x0, x1, y0, y1)

        # Create unique path for crop TFL
        if result_template[C.IS_IGNORE]:
            tag = 'i'
        else:
            tag = 'T' if result_template[C.IS_TRUE] else 'F'

        crop_path = f'{image_path[:-16]}_{row[C.COL]}{tag}_{row[C.SEQ]}.png'

        # Save unique path
        result_template[C.CROP_PATH] = crop_path

        # Create a DataFrame with the current result_template data
        result_row_df = pd.DataFrame(result_template, index=[index])

        # Concatenate the current row DataFrame with the existing result DataFrame
        result_df = pd.concat([result_df, result_row_df], ignore_index=True)
        if result_template[C.IS_TRUE] or not result_template[C.IS_IGNORE]: # TODO: only ignore = false
            # Extract image_path and open the image
            image_path = row[C.CROP_PATH]
            image = Image.open(C.PART_IMAGE_SET / C.IMAGES_1 / image_path)
            # Crop the image using the coordinates
            cropped_image = image.crop((x0, y0, x1, y1))
            # Save cropped image
            full_path = C.CROP_DIR / f'{image_path[:-4]}_{row[C.COL]}{tag}_{index}.png'
            print(f"Saving to: {full_path}")
            cropped_image.save(full_path)

    # A Short function to help you save the whole thing - your welcome ;)
    save_for_part_2(result_df)
    return result_df


def create_all_crops():
    df = pd.read_csv(C.BASE_SNC_DIR/C.ATTENTION_PATH/C.ATTENTION_CSV_NAME)
    crop_df = create_crops(df)
