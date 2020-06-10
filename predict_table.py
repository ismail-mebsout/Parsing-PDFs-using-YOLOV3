#%%
import os
import copy
import camelot

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PyPDF2 import PdfFileWriter, PdfFileReader
from pdf2image import convert_from_path, convert_from_bytes
from utils.detect_func import detectTable, parameters

import argparse
#%%
def norm_pdf_page(pdf_file, pg):
    pdf_doc = PdfFileReader(open(pdf_file, "rb"))
    pdf_page = pdf_doc.getPage(pg-1)
    pdf_page.cropBox.upperLeft = (0, list(pdf_page.mediaBox)[-1])
    pdf_page.cropBox.lowerRight = (list(pdf_page.mediaBox)[-2], 0)
    return pdf_page

def pdf_page2img(pdf_file, pg, save_image=True):
    img_page = convert_from_path(pdf_file, first_page=pg, last_page=pg)[0]
    if save_image:
        img=pdf_file[:-4]+"-"+str(pg)+".jpg"
        img_page.save(img)
    return np.array(img_page)

def outpout_yolo(output):
    output=output.split("\n")
    output.remove("")

    bboxes=[]
    for x in output:
        cleaned_output=x.split(" ")
        cleaned_output.remove("")
        cleaned_output=[eval(x) for x in cleaned_output]
        bboxes.append(cleaned_output)
    
    return bboxes

def img_dim(img, bbox):
    H_img,W_img,_=img.shape
    x1_img, y1_img, x2_img, y2_img,_,_=bbox
    w_table, h_table=x2_img-x1_img, y2_img-y1_img
    return [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]

def norm_bbox(img, bbox, x_corr=0.05, y_corr=0.05):
    [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, bbox)
    x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm=x1_img/W_img, y1_img/H_img, x2_img/W_img, y2_img/H_img
    w_img_norm, h_img_norm=w_table/W_img, h_table/H_img
    w_corr=w_img_norm*x_corr
    h_corr=h_img_norm*x_corr

    return [x1_img_norm-w_corr,y1_img_norm-h_corr/2,x2_img_norm+w_corr,y2_img_norm+2*h_corr]


def bboxes_pdf(img, pdf_page, bbox, save_cropped=False):
    W_pdf=float(pdf_page.cropBox.getLowerRight()[0])
    H_pdf=float(pdf_page.cropBox.getUpperLeft()[1])

    [x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm]=norm_bbox(img, bbox)
    x1, y1 = x1_img_norm*W_pdf, (1-y1_img_norm)*H_pdf
    x2, y2 = x2_img_norm*W_pdf, (1-y2_img_norm)*H_pdf
    
    if save_cropped:
        page=copy.copy(pdf_page)
        page.cropBox.upperLeft = (x1, y1)
        page.cropBox.lowerRight = (x2, y2)
        output = PdfFileWriter()
        output.addPage(page)

        with open("cropped_"+pdf_file[:-4]+"-"+str(pg)+".pdf", "wb") as out_f:
            output.write(out_f)

    return [x1, y1, x2, y2]

#%%
def detect_tables(opt):
    pdf_file=opt.pdf_path
    pg=opt.page

    see_example=False
    img_path=pdf_file[:-4]+"-"+str(pg)+".jpg"
    pdf_page=norm_pdf_page(pdf_file, pg)
    img = pdf_page2img(pdf_file, pg, save_image=True)

    opt=parameters(img_path)
    output_detect=detectTable(opt)
    output=outpout_yolo(output_detect)


    os.remove(img_path)
    os.rmdir("outputs")

    if see_example:
            for out in output:
                [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, out)
                plt.plot([x1_img, x2_img, x2_img, x1_img, x1_img], [y1_img, y1_img, y2_img, y2_img, y1_img], linestyle='-.', alpha=0.7)
                # plt.scatter([x1_img, x2_img], [y1_img, y2_img])
            imgplot = plt.imshow(img)
            plt.savefig(pdf_file[:-4]+"-"+str(pg)+".png")


    interesting_areas=[]
    for x in output:
        [x1, y1, x2, y2]=bboxes_pdf(img, pdf_page, x)
        bbox_camelot = [
            ",".join([str(x1), str(y1), str(x2), str(y2)])
        ][0]  # x1,y1,x2,y2 where (x1, y1) -> left-top and (x2, y2) -> right-bottom in PDF coordinate space
        interesting_areas.append(bbox_camelot)

    output_camelot = camelot.read_pdf(
        filepath=pdf_file, pages=str(pg), flavor="stream", table_areas=interesting_areas
    )
    output_camelot=[x.df for x in output_camelot]
    for i,db in enumerate(output_camelot):
        db.to_excel(pdf_file[:-4]+"-"+str(pg)+"-table-"+str(i)+".xlsx")



# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default="pdfs/boeing.pdf", help="PDF path located in pdfs folder")
    parser.add_argument("--page", type=int, default=2, help="Page to parse")
    opt = parser.parse_args()
    detect_tables(opt)


# %%
