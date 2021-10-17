"""
Contains the ultimate class for extracting licences from images (LicenceExtractor).

Also a runnable script which takes the input image path as a command line argument (-i / --input-image).

Author: Lucas Fern
lucaslfern@gmail.com
"""

import cv2
import os

from .Constants import *
from .Processors import (Resizer, BinaryInvThresholder, FastDenoiser,
                         HoughLineCornerDetector, QuadrilateralExtractor)


class LicenceExtractor:
    """Takes an image filepath and extracts licences from the image."""

    def __init__(self, output_process=False, out_dir='output') -> None:
        """
        Initialises the required processors for licence extraction.

        Parameters
        ----------
        output_process: Boolean representing whether or not intermediate steps should output images.
        out_dir: The directory to output images from intermediate steps to.
        """

        # Resizes the image to be within a specified maximum height
        self._resizer = Resizer(
            height=MAX_IMG_HEIGHT,
            output_process=output_process,
            out_dir=out_dir
        )
        # Removes noise from the input image
        self._denoiser = FastDenoiser(
            strength=DENOISER_STRENGTH,
            output_process=output_process,
            out_dir=out_dir
        )
        # Converts a colour image to black and white using a thresholding method based on absolute colour value.
        # Can substitute other thresholding methods, OTSU is worth considering.
        self._thresholder = BinaryInvThresholder(
            output_process=output_process,
            out_dir=out_dir
        )
        # Detects the straight lines in a thresholded image and returns the intersections between them.
        # These intersections are likely to represent corners of licences in the original image.
        self._corner_detector = HoughLineCornerDetector(
            thresh=HOUGH_THRESHOLD,
            output_process=output_process,
            out_dir=out_dir
        )
        # Uses the set of intersections to extract a set of quadrilaterals which are most likely to contain licences.
        self._quadrilateral_extractor = QuadrilateralExtractor(
            output_process=output_process,
            out_dir=out_dir
        )

    def __call__(self, image_path, max_images) -> list[dict]:
        """
        Attempts to extract all licenses from a PNG image.

        Parameters
        ----------
        image_path: The filepath to the image.
        max_images: The maximum amount of images to be returned/concurrently kept in memory.

        Returns
        -------
        A list of dictionaries where each dictionary contains:
          'img': The colour image of the extracted licence.
          'img_score': A score representing the likelihood that the extracted licence is a true positive.
                       Higher is better.
          'vertices': The 4 vertices bounding the extracted licence in the original image.
          'dist': The sum of the distances between the proposed locations of the bottom corners and the nearest
                  actual intersection (see Processors.QuadrilateralExtractor._find_quadrilaterals()).
        """

        # Read image from file
        self._image = cv2.imread(image_path)

        # Resize the image to within the maximum height
        self._processed = self._resized = self._resizer(self._image)

        # De-noise the image
        self._processed = self._denoiser(self._processed)

        # Generate a B+W image by thresholding
        self._processed = self._thresholder(self._processed)

        # Find corners of possible quadrilaterals in thresholded image
        intersections = self._corner_detector(self._processed, MAX_INTERSECTIONS)

        # Return extracted images.
        return self._quadrilateral_extractor(self._processed,
                                             self._resized,
                                             intersections,
                                             max_images)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Python script to detect and extract documents.")
    parser.add_argument(
        '-i',
        '--input-image',
        help="Image containing the document",
        required=True,
        dest='input_image'
    )
    args = parser.parse_args()

    # Ensure that the required output directory exists
    img_filename = args.input_image.split('/')[-1]
    out_dir = IMG_OUT_DIR + '/' + img_filename
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Extract licences from the image
    licence_extractor = LicenceExtractor(output_process=True, out_dir=out_dir)
    extracted_imgs: list[dict] = licence_extractor(args.input_image, MAX_IMAGES)

    # Save the extracted licences
    for img_dict in extracted_imgs:
        mean = img_dict['img_score']
        cv2.imwrite(f'{out_dir}/result{mean:.2f}.png', img_dict['img'])


if __name__ == "__main__":
    IMG_OUT_DIR = r'output'
    main()
