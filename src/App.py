"""
A very mediocre looking GUI for testing the Document Extraction and Field Matching algorithms.
Made for the demonstration on Oct 15 2021.

Author: Lucas Fern
lucaslfern@gmail.com
"""

import PySimpleGUI as sg
import os
import cv2
import glob
from PIL import Image
from io import BytesIO

from FieldMatching.FieldExtraction import get_matches
from DocumentExtraction.PageExtractor import LicenceExtractor


class App:
    """
    The whole GUI. Very fragile but works as an example.
    May contain some unused attributes since I adapted this from a previous project.
    """

    # Directories for the results and debug outputs
    OUT_DIR = r'..\output'
    OUT_PROCESS_DIR = r'..\process-output'

    # The fields we attempt to match from the licences
    FIELDS = ['Name',
              'Capability',
              'Provider',
              'Issue Date',
              'Expiry Date',
              'Date of Birth',
              'Licence Number',
              'Card Number',
              'Issuer',
              'Notes']

    # Config for the GUI image display
    PDF_SIZE = (595, 842)
    MAX_IMAGE_DIM = MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT = (535, 758)

    # A GUI theme with some lucidity colors
    my_new_theme = {'BACKGROUND': '#E4E5E6',
                    'TEXT': '#000000',
                    'INPUT': '#FFFFFF',
                    'TEXT_INPUT': '#000000',
                    'SCROLL': '#c7e78b',
                    'BUTTON': ('white', '#F68F1E'),
                    'PROGRESS': ('#01826B', '#D0D0D0'),
                    'BORDER': 0,
                    'SLIDER_DEPTH': 0,
                    'PROGRESS_DEPTH': 0}

    sg.theme_add_new('MyNewTheme', my_new_theme)
    sg.theme('MyNewTheme')

    # The GUI columns
    left_col = [[sg.Text('Input PDF'), sg.In(enable_events=True, key='-IN FILE-', size=(25, 1)), sg.FileBrowse()],
                [sg.Button('Begin Extraction', key='-BEGIN-')],
                [sg.Text('_' * 40)],
                [sg.Text('Extracted Documents')],
                [sg.Listbox(values=[], enable_events=True, size=(35, 20), key='-FILE LIST-',
                            select_mode=sg.LISTBOX_SELECT_MODE_SINGLE)],
                [sg.Button('Clear Output Directory', key='-CLEAR-')]]

    images_col = [[sg.Text(size=(40, 1), key='-TOUT-')],
                  [sg.Graph(canvas_size=MAX_IMAGE_DIM,
                            graph_bottom_left=(0, -MAX_IMAGE_HEIGHT),
                            graph_top_right=(MAX_IMAGE_WIDTH, 0),
                            enable_events=True,
                            key="-GRAPH-")]]

    right_col = [[sg.Text('Name', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Name', text_color='black', background_color='white')],
                 [sg.Text('Capability', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Capability', text_color='black', background_color='white')],
                 [sg.Text('Provider', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Provider', text_color='black', background_color='white')],
                 [sg.Text('Issue Date', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Issue Date', text_color='black', background_color='white')],
                 [sg.Text('Expiry Date', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Expiry Date', text_color='black', background_color='white')],
                 [sg.Text('Date of Birth', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Date of Birth', text_color='black', background_color='white')],
                 [sg.Text('Licence Number', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Licence Number', text_color='black', background_color='white')],
                 [sg.Text('Card Number', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Card Number', text_color='black', background_color='white')],
                 [sg.Text('Issuer', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Issuer', text_color='black', background_color='white')],
                 [sg.Text('Notes', size=(14, 1)),
                  sg.InputText(size=(25, 1), key='Notes', text_color='black', background_color='white')]]

    # Composing the GUI layout
    layout = [[sg.Column(left_col, element_justification='c', vertical_alignment='center'), sg.VSeperator(),
               sg.Column(images_col, element_justification='c', vertical_alignment='center'), sg.VSeperator(),
               sg.Column(right_col, element_justification='l', vertical_alignment='center')]]

    def __init__(self):
        font = ('Arial', 14)

        self.window = sg.Window('Document Detection and OCR Demonstration', App.layout, resizable=True,
                                return_keyboard_events=True,
                                font=font,
                                margins=(50, 50),
                                finalize=True)

        self.current_filename = None
        self.current_image = None
        self.current_ratio = None

        self.licence_extractor = LicenceExtractor(output_process=True, out_dir=App.OUT_PROCESS_DIR)

        self._running = True
        self.event_loop()

    def event_loop(self):
        self.window['-FILE LIST-'].update(os.listdir(App.OUT_DIR))

        while self._running:
            event, values = self.window.read()

            print(event, values)

            if event == sg.WIN_CLOSED or event == 'Exit':
                self._running = False
                break  # Runs into window.close()

            elif event == '-IN FILE-':  # Folder name was filled in, make a list of files in the folder
                self.image_selected(values['-IN FILE-'])

            elif event == '-BEGIN-':  # Perform licence extraction on the input
                extracted_imgs = self.licence_extractor(values['-IN FILE-'], max_images=20)
                for img_dict in extracted_imgs:
                    mean = img_dict['img_score']
                    cv2.imwrite(f'{App.OUT_DIR}/result{mean:.2f}.png', img_dict['img'])

                self.window['-FILE LIST-'].update(os.listdir(App.OUT_DIR))  # Put the output files in the UI list

            elif event == '-FILE LIST-':  # A file was chosen from the listbox
                for key in App.FIELDS:
                    self.window[key].update('')

                self.current_filename = values['-FILE LIST-'][0]
                self.image_selected(os.path.join(App.OUT_DIR, self.current_filename))

                self.extract_fields(os.path.join(App.OUT_DIR, self.current_filename))

            elif event == '-CLEAR-':
                for f in glob.glob(f'{App.OUT_DIR}/*') + glob.glob(f'{App.OUT_PROCESS_DIR}/*'):
                    os.remove(f)

                self.window['-FILE LIST-'].update(os.listdir(App.OUT_DIR))  # Put the output files in the UI list

        self.window.close()

    def image_selected(self, filepath):
        """Display the image. Catch errors very poorly."""
        try:
            self.window['-TOUT-'].update(filepath)
            self.current_image = Image.open(filepath)
            image, self.current_ratio = App.resize(self.current_image)
            self.window['-GRAPH-'].Erase()
            self.window['-GRAPH-'].DrawImage(data=App.convert_to_bytes(image), location=(0, 0))
            self.window.refresh()
        except Exception as E:
            print(f'** Error {E} **')
            pass  # something weird happened making the full filename

    def extract_fields(self, image_dir):
        """Gets the extracted text and populates the text fields."""
        result = get_matches(image_dir)

        for key in App.FIELDS:
            if res := result[key.lower()]:
                self.window[key].update(res)

    @staticmethod
    def resize(image):
        width, height = image.size
        if (ratio := min(App.MAX_IMAGE_WIDTH / width, App.MAX_IMAGE_HEIGHT / height)) < 1:
            image = image.resize((int(width * ratio), int(height * ratio)))
        else:
            ratio = 1

        return image, ratio

    @staticmethod
    def convert_to_bytes(image):
        with BytesIO() as bio:
            image.save(bio, format="PNG")
            del image
            return bio.getvalue()


if __name__ == '__main__':
    watermarking = App()
