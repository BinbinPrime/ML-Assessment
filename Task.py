# coding=utf-8
import xlrd
import xlwt
import string
import numpy as np
import os


class OperExcel():
    def rExcel(self, inEfile, strfilename, outfile):
        rfile = xlrd.open_workbook(inEfile)
        table = rfile.sheet_by_index(0)
        nrows = table.nrows - 1
        ncols = table.ncols

        stationsheet = xlrd.open_workbook('D:\pythonProject2\clinical_dataset.xlsx')
        stationtable = stationsheet.sheet_by_index(0)
        nstnrows = stationtable.nrows - 1

        wb = xlwt.Workbook()
        ws = wb.add_sheet('Age')