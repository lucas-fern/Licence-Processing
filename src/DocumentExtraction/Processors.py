"""
A collection of classes which are used in the licence extraction pipeline.

This code is based off https://github.com/Shakleen/Python-Document-Detector
but with significant modification.

Author: Lucas Fern
lucaslfern@gmail.com
"""

import cv2
from math import sin, cos
import numpy as np
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
import scipy.cluster.hierarchy as hcluster
from shapely.geometry import Polygon

import random

from .Constants import *


class Resizer:
    """
    Resizes image.

    Params
    ------
    image   is the image to be resized
    height  is the height the resized image should have. Width is changed by similar ratio.

    Returns
    -------
    Resized image
    """

    def __init__(self, height=1280, output_process=False, out_dir='output'):
        self._height = height
        self.output_process = output_process
        self.out_dir = out_dir

    def __call__(self, image):
        # Image height within limit
        if image.shape[0] <= self._height:
            return image

        ratio = round(self._height / image.shape[0], 3)
        width = int(image.shape[1] * ratio)
        dim = (width, self._height)

        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/resized.jpg', resized)

        return resized


class BinaryInvThresholder:
    """
    Thresholds image using the Binary Inverse method.

    Potential to use other methods, especially worth considering OTSU from OpenCV

    Params
    ------
    :image: the image to be Thresholded

    Returns
    -------
    Thresholded image
    """

    def __init__(self, thresh1=245, thresh2=255, output_process=False, out_dir='output'):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        self.out_dir = out_dir

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(image, self.thresh1, self.thresh2, cv2.THRESH_BINARY_INV)

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/thresholded.jpg', thresholded)

        return thresholded


class FastDenoiser:
    """
    Denoises image by using the fastNlMeansDenoising method

    Params
    ------
    :image: is the image to be denoised
    :strength: the amount of denoising to apply

    Returns
    -------
    Denoised image
    """

    def __init__(self, strength=7, output_process=False, out_dir='output'):
        self._strength = strength
        self.output_process = output_process
        self.out_dir = out_dir

    def __call__(self, image):
        temp = cv2.fastNlMeansDenoising(image, h=self._strength)

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/denoised.jpg', temp)
        return temp


class Closer:
    """
    Attempts to close any holes in the thresholded image by dilating the white pixels, then eroding.

    See an explanation at:
    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/
        py_morphological_ops/py_morphological_ops.html#closing

    Uses an elliptical kernel.

    Params
    ------
    :image: the image to be closed.
    :kernel_size: the size of the elliptical kernel.
    :iterations: the number of iterations of dilation/erosion.

    Returns
    -------
    Closed image
    """

    def __init__(self, kernel_size=3, iterations=10, output_process=False, out_dir='output'):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process
        self.out_dir = out_dir

    def __call__(self, image):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._kernel_size, self._kernel_size)
        )
        closed = cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self._iterations
        )

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/closed.jpg', closed)

        return closed


class EdgeDetector:
    """
    Uses Canny edge detection to find the outline of the thresholded shapes.

    Params
    ------
    :image: is the image to find the edges in
    :thresh1: and :thresh2: the lower and upper bounds for edge rejection in the hysteresis procedure of the Canny
        algorithm. Doesn't really need to make sense. Determined empirically.

    Returns
    -------
    Image containing detected edges
    """

    def __init__(self, output_process=False, out_dir='output'):
        self.output_process = output_process
        self.out_dir = out_dir

    def __call__(self, image, thresh1=50, thresh2=150, aperture_size=3):
        edges = cv2.Canny(image, thresh1, thresh2, apertureSize=aperture_size)

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/edges.jpg', edges)
        return edges


class HoughLineCornerDetector:
    """
    Uses the Hough Line detection algorithm to find straight lines in an image.

    1. Uses the Closer and EdgeDetector to generate an image containing edges from a thresholded image.
    2. Finds Hough lines along the edges.
    3. Finds intersections between the lines.

    Params
    ------
    :image: a thresholded image.
    :max_intersections: the maximum amount of intersections to be returned between detected lines.
        Important since for N intersections the QuadrilateralExtractor checks ≈ (N choose 2)/2 quadrilaterals
    :rho_acc:, :theta_acc:, :thresh: parameters for the Hough line transform.
        https://docs.opencv.org/4.5.3/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a

    Returns
    -------
    A list of 2-tuple coordinates representing the identified intersections.
    """

    def __init__(self, rho_acc=1, theta_acc=180, thresh=60, output_process=True, out_dir='output'):
        self.rho_acc = rho_acc
        self.theta_acc = theta_acc
        self.thresh = thresh
        self.output_process = output_process
        self._preprocessor = [
            Closer(output_process=output_process, out_dir=out_dir),
            EdgeDetector(output_process=output_process, out_dir=out_dir)
        ]
        self.out_dir = out_dir

    def __call__(self, image, max_intersections=100):
        # Step 1: Process for edge detection
        self._thresh_image = self._image = image

        for processor in self._preprocessor:
            self._image = processor(self._image)

        # Step 2: Get hough lines
        self._lines = self._get_hough_lines()

        # Step 3: Get intersection points
        return self._get_clustered_intersections()[:max_intersections]

    def _get_hough_lines(self) -> np.array:
        """
        Finds the lines along the straight edges in an image.

        Returns a numpy array of (r, θ) pairs specifying the lines.
        """
        lines = cv2.HoughLines(
            self._image,
            self.rho_acc,
            np.pi / self.theta_acc,
            self.thresh
        )

        # Add lines for the edges of the document. Safeguards against the case where no edges are found, and ensures
        # that we can still extract licenses when they are uploaded pre-cropped.
        height, width = self._image.shape
        image_edges = np.array([  # Edges specified as tangents to a radius from the origin, coordinates are (r, θ)
            [[0, 0]],
            [[0, np.pi / 2]],
            [[width, 0]],
            [[height, np.pi / 2]]
        ])

        if lines is not None:  # Some lines were detected
            lines = np.concatenate((image_edges, lines), axis=0)
        else:  # No lines detected - possibly thresholding resulted in all white
            lines = image_edges

        if self.output_process:
            hough_line_img = HoughLineCornerDetector.draw_hough_lines(self._get_color_image(), lines)
            cv2.imwrite(f'{self.out_dir}/hough_line.jpg', hough_line_img)

        return lines

    @staticmethod
    def draw_hough_lines(image, lines):
        """Annotates an image with the provided lines for debugging purposes."""

        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            n = 5000  # The length of the line we draw in the image.
            x1 = int(x0 + n * (-b))
            y1 = int(y0 + n * a)
            x2 = int(x0 - n * (-b))
            y2 = int(y0 - n * a)

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        return image

    def _get_clustered_intersections(self):
        """Finds the intersections between groups of lines."""
        lines = self._lines
        intersections = []
        group_lines = combinations(range(len(lines)), 2)

        for i, j in group_lines:
            line_i, line_j = lines[i][0], lines[j][0]

            # If the angle between the lines is within ±10 degrees of a right angle...
            if 80.0 < self._get_angle_between_lines(line_i, line_j) < 100.0:
                int_point = self._intersection(line_i, line_j)

                # ...and the intersection between the lines is within the border of the image...
                if (0 <= int_point[0] <= self._image.shape[1]
                        and 0 <= int_point[1] <= self._image.shape[0]):

                    # ...then record the intersection.
                    intersections.append(int_point)

        # Perform clustering on the intersections to eliminate duplicates when close together.

        # print(f'pre cluster {len(intersections)=}')
        intersections = self._cluster_intersections(intersections)
        # print(f'post cluster {len(intersections)=}')

        if self.output_process:
            self._draw_intersections(intersections)

        return intersections

    def _cluster_intersections(self, intersections):
        """
        Perform clustering on the intersections to eliminate duplicates when intersections exist within a small
        radius of each other.
        """
        def centroid(coords):
            return tuple(int(j) for j in coords.mean(axis=0))

        # Get the cluster labels of each intersection
        clusters = hcluster.fclusterdata(intersections, MIN_VERTEX_DISTANCE, criterion='distance')

        # Replace find the centroid of each cluster and use this as the new intersection location.
        centroids = []
        intersections = np.array(intersections)
        for i in range(min(clusters), max(clusters) + 1):
            points_in_i = intersections[clusters == i]
            centroids.append(centroid(points_in_i))

        return centroids

    def _get_angle_between_lines(self, line_1, line_2):
        """
        Gets the angle in degrees between two lines in (r, θ) form. You should check this method out in the
        original code I adapted. Hilariously bad.

        (Line 146: https://github.com/Shakleen/Python-Document-Detector/blob/master/hough_line_corner_detector.py)
        """
        _, theta1 = line_1
        _, theta2 = line_2

        return abs(theta2 - theta1) * (180 / np.pi)

    def _intersection(self, line1, line2):
        """
        Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2

        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])

        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))

        return x0, y0

    def _draw_intersections(self, intersections):
        """Draws the set of intersections to an image for debugging purposes."""
        # Get the Hough Line image to annotate with the intersections
        hough_img = HoughLineCornerDetector.draw_hough_lines(self._get_color_image(), self._lines)

        for point in intersections:
            x, y = point

            cv2.circle(
                hough_img,
                (x, y),
                5,
                (255, 255, 127),
                5
            )

        cv2.imwrite(f'{self.out_dir}/intersection_point_output.jpg', hough_img)

    def _get_color_image(self):
        return cv2.cvtColor(self._thresh_image.copy(), cv2.COLOR_GRAY2RGB)


class QuadrilateralExtractor:
    """
    Takes a thresholded image, colour image, and set of intersections, and attempts to extract quadrilaterals from the
    intersections which are most likely to contain licences in the colour image. Takes a parameter max_internal_imgs
    as an upper bound on the amount of images which will be loaded into memory concurrently.
    """
    def __init__(self, output_process=False, out_dir='output'):
        self.output_process = output_process
        self.out_dir = out_dir

        self._shape_extractor = ShapeExtractor(
            output_process=output_process,
            out_dir=out_dir
        )

    def __call__(self, thresh_img, colour_img, intersections, max_internal_imgs):
        self._thresh_img = thresh_img  # The thresholded image
        self._colour_img = colour_img  # The original image
        self._intersections = intersections

        quads = self._find_quadrilaterals()
        img_dicts = self._best_quadrilaterals(quads, max_internal_imgs)
        img_dicts = self._remove_overlapping_quads(img_dicts)

        return self._get_colour_imgs(img_dicts)

    def _find_quadrilaterals(self) -> list[dict]:
        """
        Looks at all pairs of intersections found in an image and attempts to find a quadrilateral which uses the
        vector between these points as the top edge. Estimates where the bottom corners should be and finds the nearest
        intersection to the estimated location. Records the 4 corners of each quadrilateral as well as how far the
        bottom corners were from the estimated locations.

        Returns
        -------
        A list of dictionaries where each contains:
          'vertices': The 4 vertices of the quadrilateral
          'dist': The sum of the distances between the proposed locations of the bottom corners and the nearest
                  actual intersection.
        """

        ints = self._intersections
        quads = []

        # Fit a Nearest Neighbours classifier to the set of intersections.
        # Helps us to find the nearest point to a given coordinate later.
        neigh = NearestNeighbors().fit(ints)

        # Iterate over each pair of intersection points. Consider these to be the top left and right of a quadrilateral
        for tl, tr in combinations(ints, 2):
            # If top left isn't to the left of top right then swap them.
            if tl[0] > tr[0]:
                tl, tr = tr, tl

            tl = np.array(tl)
            tr = np.array(tr)

            # If our 'top edge' is more vertical than horizontal, then reject it, we want horizontal(-ish) lines.
            dx, dy = hor_edge = tr - tl
            if dy > dx or np.linalg.norm(tr - tl) < MIN_CARD_WIDTH:
                continue

            # Get a vector representing the approximate length and direction of a vertical edge by taking the top edge,
            # rotating it via. matrix multiplication, then scaling it according to the aspect ratio of a card.
            theta = np.pi / 2
            rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            vert_edge = (rot @ hor_edge) / CARD_ASPECT_RATIO

            # Get the coordinates of the bottom corners of the card.
            bl = tl + vert_edge
            br = tr + vert_edge

            # Use the NN classifier to find the nearest point to bl and br and record the distance to the nearest point
            dist_bl, idx_bl = neigh.kneighbors([bl], n_neighbors=1)
            dist_bl = dist_bl[0][0]
            idx_bl = idx_bl[0][0]
            bl = list(ints[idx_bl])

            dist_br, idx_br = neigh.kneighbors([br], n_neighbors=1)
            dist_br = dist_br[0][0]
            idx_br = idx_br[0][0]
            br = list(ints[idx_br])

            # Reject any cards which dont meet the minimum width and height (in pixels)
            if tr[0] - tl[0] < MIN_CARD_WIDTH \
                    or br[0] - bl[0] < MIN_CARD_WIDTH \
                    or bl[1] - tl[1] < MIN_CARD_WIDTH / CARD_ASPECT_RATIO \
                    or br[1] - tr[1] < MIN_CARD_WIDTH / CARD_ASPECT_RATIO:
                continue

            # Record the quadrilateral vertices as well as the distance from the bottom corners to their predicted loc.
            vertices = (tl.tolist(), tr.tolist(), br, bl)
            dist = dist_bl + dist_br
            quads.append({'dist': dist, 'vertices': vertices})

        # Return the list of quadrilateral dicts ordered in increasing bottom corner distance.
        quads.sort(key=lambda x: x['dist'])
        return quads

    def _best_quadrilaterals(self, quads: list[dict], max_quads: int) -> list[dict]:
        """
        Takes the output from _find_quadrilaterals() and scores each quadrilateral based on the average colour value
        in that quadrilateral in the thresholded image. A good candidate card should have a high average pixel value
        since the thresholded image will be mostly white. Conversely a bad quadrilateral will contain black in the
        thresholded image.

        Returns
        -------
        A list of dictionaries with length less than or equal to max_quads.
        Each dictionary contains:
          'img_score': A score for the image derived from the mean

          Plus the result passed in from _find_quadrilaterals():
            'vertices': The 4 vertices of the quadrilateral
            'dist': The sum of the distances between the proposed locations of the bottom corners and the nearest
                    actual intersection.
        """
        quad_imgs = []

        for quad in quads:
            # Get the relevant section of the thresholded image and calculate its score
            img = self._shape_extractor(self._thresh_img, quad['vertices'])
            quad['img_score'] = img.mean() / (quad['dist'] + DIST_LENIENCY)

            quad_imgs.append(quad)

            # Remove the candidates with the lowest scores if we have more than the specified maximum amount.
            quad_imgs.sort(key=lambda x: x['img_score'], reverse=True)
            while len(quad_imgs) > max_quads:
                quad_imgs.pop()

        if self.output_process:
            self._draw_quadrilaterals(quad_imgs)

        return quad_imgs

    def _remove_overlapping_quads(self, img_dicts: list[dict]) -> list[dict]:
        """
        Takes a list of dicts formatted from _best_quadrilaterals() and searches for pairs which have significant
        overlap. When significant overlap is found we discard the candidate with the lowest score.

        Returns
        -------
        A list of dictionaries with the same format as _best_quadrilaterals() but with overlapping entries eliminated.
        """
        discards = []
        for d1, d2 in combinations(img_dicts, 2):
            p1 = Polygon(d1['vertices'])
            p2 = Polygon(d2['vertices'])

            # Find the overlap between quadrilaterals
            intersection_area = p1.intersection(p2).area

            # Find how much of each quad is overlapping with the other
            prop1 = intersection_area / p1.area
            prop2 = intersection_area / p2.area

            if prop1 > OVERLAP_THRESHOLD or prop2 > OVERLAP_THRESHOLD:
                # If p1 contains more "card" then discard p2
                if d1['img_score'] > d2['img_score']:
                    discards.append(d2)
                else:  # In the other case, discard p1
                    discards.append(d1)

        return [i for i in img_dicts if i not in discards]

    def _get_colour_imgs(self, img_dicts):
        """
        Adds the colour images to a list of dictionaries formatted like the output from _remove_overlapping_quads().

        Returns
        -------
        A list of dictionaries with all the fields from the output of _remove_overlapping_quads() and an additional
        field 'img' which contains the colour image object corresponding to the area of the original image bounded
        by the quadrilateral.
        """

        for img_dict in img_dicts:
            img_dict['img'] = self._shape_extractor(self._colour_img, img_dict['vertices'])

        return img_dicts

    def _draw_quadrilaterals(self, img_dicts):
        """Draws the quadrilaterals from an img_dict onto the original image. Labels each with their score."""
        colours = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0),
                   (0, 0, 255), (75, 0, 130), (238, 130, 238)]
        grouped_output = self._colour_img.copy()

        for img_dict in img_dicts:
            tl, tr, br, bl = img_dict['vertices']

            rgb = random.choice(colours)

            for (x1, y1), (x2, y2) in [(tl, tr), (tr, br), (br, bl), (bl, tl)]:
                cv2.line(
                    grouped_output,
                    (x1, y1),
                    (x2, y2),
                    rgb,
                    2
                )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1

            text_coord = (tl[0], tl[1] + 50)
            cv2.putText(grouped_output, str(round(img_dict['img_score'], 2)), text_coord, font,
                        font_scale, rgb, 1, cv2.LINE_AA)

        cv2.imwrite(f'{self.out_dir}/grouped.jpg', grouped_output)


class ShapeExtractor:
    def __init__(self, output_process=False, out_dir='output'):
        self.output_process = output_process
        self.out_dir = out_dir

    def __call__(self, image, corners):
        # obtain a consistent order of the points and unpack them individually
        pts = np.array(corners)
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_bottom), int(width_top))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_right), int(height_left))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],  # Top left point
            [max_width - 1, 0],  # Top right point
            [max_width - 1, max_height - 1],  # Bottom right point
            [0, max_height - 1]],  # Bottom left point
            dtype="float32"  # Data type
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        if self.output_process:
            cv2.imwrite(f'{self.out_dir}/deskewed.jpg', warped)

        # return the warped image
        return warped

    def _order_points(self, pts):
        """
        Function for getting the bounding box points in the correct
        order

        Params
        pts     The points in the bounding box. Usually (x, y) coordinates

        Returns
        rect    The ordered set of points
        """
        # initialzie a list of coordinates that will be ordered such that 
        # 1st point -> Top left
        # 2nd point -> Top right
        # 3rd point -> Bottom right
        # 4th point -> Bottom left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect
