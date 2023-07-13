import os
import sys
import tkinter as tk
from tkinter import *
from tkinter import filedialog

import matplotlib
import matplotlib.pyplot as plt
import methcomp
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from docx.shared import Mm
from docx2pdf import convert
from docxtpl import DocxTemplate, InlineImage
from methcomp import passingbablok, blandaltman
from scipy import stats
from tkPDFViewer import tkPDFViewer as pdf

sns.set(style="whitegrid")
doc = DocxTemplate("./validation_template.docx")

# root = Tk()
# root.title("Report Generator")
# root.geometry("1000x1000")
# # root.configure(bg='white')


# def resource_path(relative_path):
#     if hasattr(sys, '_MEIPASS'):
#         return os.path.join(sys._MEIPASS, relative_path)
#     return os.path.join(os.path.abspath("."), relative_path)
#
#
# doc = DocxTemplate(resource_path(r"validation_template.docx"))


def comparison_data_to_context(context, series1, series2, name1, name2):
    context['name_series1'] = name1
    context['name_series2'] = name2
    values_list = []
    for i in range(len(series1.array)):
        values_list.append([series1.array[i], series2.array[i]])
    context['comparison_values'] = values_list


def boxplot_to_context(context, list_of_series, basename, width=Mm(150)):
    i = 1
    for series in list_of_series:
        name = '{}{}'.format(basename, i)
        context[name] = InlineImage(doc, series.boxplot.filepath, width=width)
        i += 1


class Image:
    def __init__(self, img, filepath):
        self.img = img
        self.filepath = filepath


class Series:
    def __init__(self, array, path, target=None):
        self.array = array.dropna()
        self.path = path
        self.boxplot = self.fboxplot(self.array)
        self.target = target
        self.samplesize = len(self.array)
        self.min = min(self.array)
        self.max = max(self.array)
        self.mean = round(self.fmean(self.array), 4)
        self.median = round(self.fmedian(self.array), 4)
        self.var = round(self.fvar(self.array), 4)  # Variance
        self.std = round(self.fstd(self.array), 4)  # Standard deviation
        self.cv = self.fcv(self.array)  # Coefficient of variation
        self.cv_percent = '{}%'.format(round(self.cv * 100, 2))
        self.nt = round(self.fnt(self.array), 4)
        self.bias = self.fbias(self.array, self.target)
        self.mu = self.fmu(self.array, self.target)

    def __str__(self):
        return ("Values: {}".format(list(self.array)) + "\nTarget value: {}".format(self.target)
                + "\nSample size: {}".format(self.samplesize)
                + "\nMinimum: {}".format(self.min) + "\nMaximum: {}".format(self.max) + "\nMean: {}".format(self.mean)
                + "\nMedian: {}".format(self.median) + "\nVariance: {}".format(self.var)
                + "\nStandard deviation: {}".format(self.std) + "\nCoefficient of variation: {} ({}%)".format(self.cv,
                                                                                                              self.cv *
                                                                                                              100)
                + "\nD'Agostino-Pearson test for Normal distribution: {}".format(self.nt)
                + "\nBias: {}".format(self.bias) + "\nMeasurement uncertainty: {}".format(self.mu))

    @classmethod
    def create_series(cls, array, path, target=None):
        if array.dropna().any():
            return cls(array, path, target)
        return None

    def fboxplot(self, array, figsize=(12, 3), dpi=300, **kwargs):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_title('Boxplot')
        ax.boxplot(array, vert=False, labels=[array.name])

        filetype = 'png'
        # path = 'Figures'
        os.makedirs(self.path, exist_ok=True)
        filepath = '{}/{}_boxplot.{}'.format(self.path, self.array.name, filetype)
        filepath = os.path.normpath(filepath)
        fig.savefig(filepath)
        return Image(fig, filepath)

    def fmean(self, array):
        return np.mean(array)

    def fmedian(self, array):
        return np.median(array)

    def fvar(self, array):
        return np.var(array, ddof=1)

    def fstd(self, array):
        return np.std(array, ddof=1)

    def fcv(self, array):
        return np.std(array, ddof=1) / np.mean(array)

    def fnt(self, array):
        return stats.normaltest(array)[1]

    def fbias(self, array, target):
        if target:
            return round(100 - (100 * target / self.fmean(array)), 4)
        return None

    def fmu(self, array, target, k=2):
        """
        Calculates and returns the measurement uncertainty ("Messunsicherheit").
        k = 2 for expanded uncertainty ("erweiterte Messunsicherheit")
        """
        if target:
            return round(k * np.sqrt((100 * self.fcv(array)) ** 2 + self.fbias(array, target) ** 2), 4)
        return None


class Correlation:
    def __init__(self, array1, array2, method):
        r, p, lo, hi = self.regression_ci(array1, array2, method)
        self.r = round(r, 4)
        self.p = p
        self.ci_lo = round(lo, 4)
        self.ci_hi = round(hi, 4)

    def regression_ci(self, x, y, method='spearman', alpha=0.05):
        """ calculate Pearson correlation along with the confidence
        interval using scipy and numpy

        Parameters
        ----------
        x, y : iterable object such as a list or np.array
          Input for correlation calculation
        method : string to designate correlation method
          'spearman' for Spearman correlation coefficient
          'pearson' for Pearson correlation coefficient
          'kendall' for Kendall's tau
        alpha : float
          Significance level. 0.05 by default

        Returns
        -------
        r : float
          Pearson's correlation coefficient
        pval : float
          The corresponding p value
        lo, hi : float
          The lower and upper bound of confidence intervals
        """

        def get_method(method):
            if method == 'spearman':
                return stats.spearmanr
            elif method == 'pearson':
                return stats.pearsonr
            elif method == 'kendall':
                return stats.kendalltau

        r, p = get_method(method)(x, y)
        r_z = np.arctanh(r)
        se = 1 / np.sqrt(x.size - 3)
        z = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = r_z - z * se, r_z + z * se
        lo, hi = np.tanh((lo_z, hi_z))
        return r, p, lo, hi


class Comparison:
    def __init__(self, array1, array2, path, name1='Method 1',
                 name2='Method 2'):
        self.path = path
        self.cor = self.fcor(array1, array2)
        self.pearson = Correlation(array1, array2, method='pearson')
        self.spearman = Correlation(array1, array2, method='spearman')
        self.kendall = Correlation(array1, array2, method='kendall')
        self.pb = self.fpb(array1, array2, name1, name2)
        self.ba = self.fba(array1, array2)

    def fcor(self, array1, array2):
        # Pearson correlation coefficient r, Significance level p
        r, p = stats.pearsonr(array1, array2)
        if p < 0.0001:
            p = '<0.0001'
        return r, p

    def fpb(self, array1, array2, name1, name2, figsize=(12, 8), dpi=300):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        passingbablok(array1, array2, CI=.95, x_label=name1, y_label=name2)

        filetype = 'png'
        # path = 'Figures'
        os.makedirs(self.path, exist_ok=True)
        filepath = '{}/passing-bablok.{}'.format(self.path, filetype)
        filepath = os.path.normpath(filepath)
        fig.savefig(filepath)
        return Image(fig, filepath)

    def fba(self, array1, array2, figsize=(12, 8), dpi=300):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        blandaltman(array1, array2, CI=.95, reference=True)

        filetype = 'png'
        # path = 'Figures'
        os.makedirs(self.path, exist_ok=True)
        filepath = '{}/bland-altman.{}'.format(self.path, filetype)
        filepath = os.path.normpath(filepath)
        fig.savefig(filepath)
        return Image(fig, filepath)


class MethodEvaluation:
    def __init__(self, df, path):
        self.method = df.Methode[0]
        self.platform = df.Plattform[0]
        self.unit = df.Einheit[0]
        self.module = df.Module[0]
        self.material = df.Material[0]
        self.name_old = df.iloc[:, 1:3].dropna().iloc[:, 0].name
        self.series_old = Series(df.iloc[:, 1:3].dropna().iloc[:, 0], path)
        self.name_new = df.iloc[:, 1:3].dropna().iloc[:, 1].name
        self.series_new = Series(df.iloc[:, 1:3].dropna().iloc[:, 1], path)
        self.comparison = Comparison(self.series_old.array, self.series_new.array,
                                     path, self.name_old, self.name_new)
        self.intraday = []

        names_intraday = ['Serie LEVEL 1', 'Serie LEVEL 2', 'Serie LEVEL 3']
        j = 1

        for name in names_intraday:
            target_prefix = 'Ziel LEVEL '
            target = target_prefix + str(j)
            series = Series.create_series(df[name], path, df[target].iloc[0])
            if series:
                self.intraday.append(series)
            j += 1
        self.daytoday = []
        names_daytoday = ['DtoD.LEVEL 1', 'DtoD.LEVEL 2', 'DtoD.LEVEL 3']
        i = 1
        for name in names_daytoday:
            target_prefix = 'Ziel LEVEL '
            target = target_prefix + str(i)
            series = Series.create_series(df[name], path, df[target].iloc[0])
            if series:
                self.daytoday.append(series)
            i += 1

        self.wbias_intraday = round(self.fwbiasmu(self.intraday)[0], 4)
        self.wmu_intraday = round(self.fwbiasmu(self.intraday)[1], 4)
        self.wbias_daytoday = round(self.fwbiasmu(self.daytoday)[0], 4)
        self.wmu_daytoday = round(self.fwbiasmu(self.daytoday)[1], 4)

    def fwbiasmu(self, list_of_series):
        # Calculate sample-size weighted mean of bias and measurement
        # uncertainty
        sum_n = sum([x.samplesize for x in list_of_series])
        w_bias = sum([(x.samplesize * x.bias) for x in list_of_series])
        w_mu = sum([x.samplesize * x.mu for x in list_of_series])

        wbias = w_bias / sum_n
        wmu = w_mu / sum_n

        return wbias, wmu


class SysInfo:
    def __init__(self):
        self.python = sys.version
        self.pandas = pd.__version__
        self.numpy = np.__version__
        self.scipy = sp.__version__
        self.methcomp = methcomp.__version__
        self.matplotlib = matplotlib.__version__
        self.seaborn = sns.__version__

    def get(self):
        return 'Python v{}; pandas v{}; NumPy v{}; SciPy v{}; ' \
               'Methcomp v{}; Matplotlib v{}; seaborn v{}.'.format(
            self.python, self.pandas, self.numpy, self.scipy,
            self.methcomp, self.matplotlib, self.seaborn)


# def extract_info(fln):
#     name_of_file_0 = os.path.basename(fln)
#     name_of_file = os.path.splitext(name_of_file_0)[0]
#     path_name = os.path.dirname(fln)
#     # print(name_of_file)
#     df = pd.read_csv(fln, sep=';')
#     path_name_2 = os.path.normpath(os.path.join(path_name, 'Figures'))

#     me = MethodEvaluation(df, path_name_2)

#     context = {
#         'me': me,
#         'old_boxplot': InlineImage(doc, me.series_old.boxplot.filepath, width=Mm(150)),
#         'new_boxplot': InlineImage(doc, me.series_new.boxplot.filepath, width=Mm(150)),
#         'passingbablok': InlineImage(doc, me.comparison.pb.filepath, width=Mm(160)),
#         'blandaltmann': InlineImage(doc, me.comparison.ba.filepath, width=Mm(160)),
#         'sysinfo': SysInfo().get(), }

#     comparison_data_to_context(context, me.series_old, me.series_new, me.name_old, me.name_new)

#     boxplot_to_context(context, me.intraday, 'intraday')
#     boxplot_to_context(context, me.daytoday, 'daytoday')

#     doc.render(context)
#     docx_name0 = os.path.join(name_of_file + ".docx")
#     docx_name = os.path.normpath(os.path.join(path_name, docx_name0))

#     pdf_name0 = os.path.join(name_of_file + ".pdf")
#     pdf_name = os.path.normpath(os.path.join(path_name, pdf_name0))

#     doc.save(docx_name)

#     convert(docx_name, pdf_name)

#     # pdf_path_name = docx_name[:-4]+'pdf'

#     # pdf_path_name = os.path.join(path_name, name_of_file + ".pdf")

#     return pdf_name

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QStatusBar, QSizePolicy
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtWebEngineWidgets import QWebEngineSettings, QWebEngineView
from PyQt5.QtGui import QFont, QColor
import time
import os

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        self.resize(900, 900)
        self.setWindowTitle('Report Generator')
        # self.setStyleSheet("background-color: #6A5ACD;")

        self.layout = QVBoxLayout(self)
        self.button_layout = QHBoxLayout()

        self.button = QPushButton("Select CSV file/s", self)
        self.button.clicked.connect(self.browse_csv)
        self.button.setMaximumWidth(250)  # Set maximum width of button
        self.button.setMinimumHeight(40)  # adjusted button height
        # self.button.setStyleSheet("background-color: #FFD700;")  # Change to Gold color
        self.button.setFont(QFont('Arial', 11))

        self.button2 = QPushButton("Show report", self)
        self.button2.clicked.connect(self.show_pdf_file)
        self.button2.setMaximumWidth(250)  # Set maximum width of button
        self.button2.setMinimumHeight(40)  # adjusted button height
        # self.button2.setStyleSheet("background-color: #FFD700;")  # Change to Gold color
        self.button2.setFont(QFont('Arial', 11))

        self.button3 = QPushButton("Exit", self)
        self.button3.clicked.connect(self.close)
        self.button3.setMaximumWidth(250)  # Set maximum width of button
        self.button3.setMinimumHeight(40)  # adjusted button height
        # self.button3.setStyleSheet("background-color: #FFD700;")  # Change to Gold color
        self.button3.setFont(QFont('Arial', 11))

        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.button)
        self.button_layout.addWidget(self.button2)
        self.button_layout.addWidget(self.button3)
        self.button_layout.addStretch(1)

        self.label = QLabel("Selected file: ", self)
        self.label.setMinimumHeight(50)
        # self.label.setStyleSheet("background-color: #FFFACD;")  # Change to LemonChiffon color
        self.label.setFont(QFont('Arial', 11))

        self.pdf_viewer = QWebEngineView(self)
        self.pdf_viewer.settings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
        self.pdf_viewer.settings().setAttribute(QWebEngineSettings.PdfViewerEnabled, True)
        self.pdf_viewer.setMinimumSize(800, 600)

        self.statusBar = QStatusBar()
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.pdf_viewer, 1)  # add stretch factor to make sure it takes up remaining space
        self.layout.addWidget(self.statusBar)


        self.progress = QProgressBar(self)
        self.layout.addWidget(self.progress)

        # self.timer = QTimer()
        # self.timer.timeout.connect(self.handleTimer)
        # self.timer.start(100)  # update progress every 100 ms

    def handleTimer(self):
        if self.progress.value() < 100:
            self.progress.setValue(self.progress.value() + 1)
        else:
            self.timer.stop()

    def extract_info(self, fln):
        self.progress.setValue(0)
        self.statusBar.showMessage("Processing CSV file...")

        name_of_file_0 = os.path.basename(fln)
        name_of_file = os.path.splitext(name_of_file_0)[0]
        path_name = os.path.dirname(fln)
        df = pd.read_csv(fln, sep=';')
        path_name_2 = os.path.normpath(os.path.join(path_name, 'Figures'))

        me = MethodEvaluation(df, path_name_2)

        self.progress.setValue(10)  # Update progress bar
        time.sleep(1)  # Simulate time-consuming process

        context = {
            'me': me,
            'old_boxplot': InlineImage(doc, me.series_old.boxplot.filepath, width=Mm(150)),
            'new_boxplot': InlineImage(doc, me.series_new.boxplot.filepath, width=Mm(150)),
            'passingbablok': InlineImage(doc, me.comparison.pb.filepath, width=Mm(160)),
            'blandaltmann': InlineImage(doc, me.comparison.ba.filepath, width=Mm(160)),
            'sysinfo': SysInfo().get(), }

        comparison_data_to_context(context, me.series_old, me.series_new, me.name_old, me.name_new)

        self.progress.setValue(30)  # Update progress bar
        time.sleep(1)  # Simulate time-consuming process

        boxplot_to_context(context, me.intraday, 'intraday')
        boxplot_to_context(context, me.daytoday, 'daytoday')

        doc.render(context)

        self.progress.setValue(60)  # Update progress bar
        time.sleep(1)  # Simulate time-consuming process

        docx_name0 = os.path.join(name_of_file + ".docx")
        docx_name = os.path.normpath(os.path.join(path_name, docx_name0))

        pdf_name0 = os.path.join(name_of_file + ".pdf")
        pdf_name = os.path.normpath(os.path.join(path_name, pdf_name0))

        doc.save(docx_name)

        convert(docx_name, pdf_name)

        self.progress.setValue(100)  # Update progress bar
        self.statusBar.clearMessage()

        return pdf_name



    def browse_csv(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open CSV file', os.getenv('HOME'), "CSV (*.csv)")
        if filename:
            self.label.setText("Selected file: " + str(filename))

            pdf_path_name = self.extract_info(filename)  # Call method

            self.pdf_viewer.load(QUrl.fromLocalFile(pdf_path_name))

    def show_pdf_file(self):
        self.statusBar.showMessage("Opening PDF file...")
        filename, _ = QFileDialog.getOpenFileName(self, 'Open PDF file', os.getenv('HOME'), "PDF (*.pdf)")
        if filename:
            self.label.setText("Selected file: " + str(filename))
            self.pdf_viewer.load(QUrl.fromLocalFile(filename))
        self.statusBar.clearMessage()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())






# def browse_csv():
#     global v1, v2, lbl

#     filenames = filedialog.askopenfilenames(initialdir=os.getcwd(),
#                                             title="Select Image file",
#                                             filetypes=(("CSV File", "*.csv"),
#                                                        ("CSV File", "*.CSV"),
#                                                        ("All files", "*.")))

#     if len(filenames) > 1:
#         for i in range(len(filenames)):
#             filename = os.path.normpath(filenames[i])
#             extract_info(filename)
#         Label(root, text="Select single PDF generated report to view using "
#                          "Show report button", fg="red").pack()
#     else:
#         filename = filenames[0]

#         if filename:
#             if v2:
#                 v2.destroy()
#             if v1:
#                 # v1.destroy()
#                 v1.img_object_li.clear()
#         tot = Label(root)
#         tot.config(fg='blue', text="Selected file: " + str(filename))
#         tot.pack(side=TOP)
#         filename = os.path.normpath(filename)

#         pdf_path_name = extract_info(filename)

#         v1 = pdf.ShowPdf()
#         v2 = v1.pdf_view(root, pdf_location=open(pdf_path_name, "r"),
#                          width=120, height=180)

#         # v1 = DocViewer(root)
#         # v2 = v1.display_file(pdf_path_name)

#         v2.pack(pady=(20, 20))


# def show_pdf_file():
#     global v1, v2, lbl

#     filename = filedialog.askopenfilename(initialdir=os.getcwd(),
#                                           title="Select Image file",
#                                           filetypes=(("PDF File", "*.pdf"),
#                                                      ("PDF File", "*.PDF"),
#                                                      ("All files", "*.")))

#     if filename:
#         if v2:
#             v2.destroy()
#         if v1:
#             # v1.destroy()
#             v1.img_object_li.clear()
#         tot = Label(root)
#         tot.config(fg='blue', text="Selected file: " + str(filename))
#         tot.pack(side=TOP)

#     v1 = pdf.ShowPdf()
#     v2 = v1.pdf_view(root, pdf_location=open(filename, "r"), width=120, height=180)

#     # v1 = DocViewer(root)
#     # v2 = v1.display_file(filename)

#     v2.pack(pady=(15, 15))


# frm = Frame(root)
# frm.pack(side=TOP, padx=10, pady=10)

# btn = Button(frm, text="Select CSV file/s", command=browse_csv)
# btn.pack(side=tk.LEFT, )

# btn2 = Button(frm, text="Show report", command=show_pdf_file)
# btn2.pack(side=tk.LEFT, padx=5)

# # btn3 = Button(frm, text="Exit", command=lambda: exit())
# # btn3.pack(side=tk.LEFT, padx=5)

# v1, v2, lbl = None, None, None

# # tk.messagebox.showinfo("ERROR", traceback.format_exc())
# root.mainloop()
