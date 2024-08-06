from pptx.util import Inches
import os
import random
from pptx import Presentation
from pptx.util import Inches, Pt
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from centralizedData import plotsFolder, presentationsFolder


MAC = 1
both = 1

columns = ['orders_per_eff_online', 'exposure_per_eff_online_b_p1p2', 'ted_gmv_r_burn_gmv_b2c_gmv_p2c_gmv',
           #    "b_cancel_rate", 'bad_rating_rate', 'imperfect_order_rate',
           'eff_online_rs', 'daily_orders', 'priorityChanges',
           'imperfect_order_rate_bad_rating_rate', 'eff_online_rs_healthy_stores',
           ]

# P1P2 vs UV en distintos ejes


def getFilesFromFolder(folder):
    files = []
    for root, dirs, file in os.walk(folder):
        for f in file:
            if '.txt' in f:
                continue
            files.append(f)
    return files


def getGoodFile(folder):
    files = getFilesFromFolder(folder)
    pool = ['BB.png', 'TB.png', 'BT.png', 'TT.png']
    for f in pool:
        if f not in files:
            return f
    pool = ['TT.png']
    return random.choice(pool)


def doGreatComment(file: str, position: tuple = None, slide=None):
    column = file.split('/')[-2]
    if position is None:
        raise ValueError('Position is required')
    with open(file, 'r') as file:
        content = file.read()
    inverse = False
    if column in ["b_cancel_rate", 'bad_rating_rate', 'imperfect_orders', 'imperfect_order_rate']:
        inverse = True
    divided = content.split('@')
    if content == 'The change cannot be calculated over the last 3 months':
        return
    if len(divided) == 1:
        add_text(slide, content, font_size=0.15,
                 bold=False, color=(0, 0, 0), position=position, vertical=False,
                 BG=(242, 242, 242))
        return
    else:
        firstPart = add_text(slide, divided[0], font_size=0.15,
                             bold=False, color=(0, 0, 0), position=position, vertical=False,
                             BG=(242, 242, 242))
        if inverse:
            if divided[1] == "a worse tendency":
                divided[1] = "a better tendency"
            else:
                divided[1] = "a worse tendency"
        if divided[1] == "a worse tendency":
            color = (204, 0, 0)
            add_text(slide, divided[1], font_size=0.15, new=False, last_p=firstPart,
                     bold=True, color=color, )
        else:
            color = (127, 169, 108)
            add_text(slide, divided[1], font_size=0.15, new=False, last_p=firstPart,
                     bold=True, color=color, )
        add_text(slide, divided[2], font_size=0.15, new=False, last_p=firstPart,
                 bold=False, color=(0, 0, 0), BG=(242, 242, 242))
    return


def add_text(slide, text_letters, new=True, last_p=None, font_size=0.5, bold=False, color=(0, 0, 0), position=None, vertical=True, BG=None):
    if position is None and new:
        raise ValueError('Position is required')

    if new:
        # Add new textbox
        text_box = slide.shapes.add_textbox(
            Inches(position[0]), Inches(position[1]), Inches(position[2]), Inches(position[3]))
        text_frame = text_box.text_frame
        text_frame.word_wrap = True
        text_frame.auto_size = False
        text_frame.text = text_letters
        if vertical:
            text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        # set font roboto
        font = text_frame.paragraphs[0].runs[0].font
        font.name = 'Roboto'
        # set background color
        if BG is not None:
            fill = text_box.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(BG[0], BG[1], BG[2])

        # Set font properties
        for paragraph in text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Inches(font_size)
                run.font.name = 'Roboto'
                run.font.bold = bold
                run.font.color.rgb = RGBColor(color[0], color[1], color[2])
                run.vertical_anchor = MSO_ANCHOR.MIDDLE
                if BG is not None:
                    fill = text_box.fill
                    fill.solid()
                    fill.fore_color.rgb = RGBColor(BG[0], BG[1], BG[2])

        # Return last paragraph for chaining
        return text_frame.paragraphs[-1]

    elif not new and last_p is not None:
        # Add text to existing paragraph
        run = last_p.add_run()
        run.text = ' ' + text_letters
        run.font.size = Inches(font_size)
        run.font.name = 'Roboto'
        run.font.bold = bold
        run.font.color.rgb = RGBColor(color[0], color[1], color[2])

        # Return updated paragraph for chaining
        return last_p

    else:
        raise ValueError('Invalid arguments')


def createFolder(folder=str):
    folder = f'{presentationsFolder}/{folder}'
    if not os.path.exists(folder):
        os.makedirs(folder)


def getImagesAndNotes(folderInitial: str, pais: str, prioridad: str):
    graph_image_paths = []
    graph_text_paths = []
    # Adjust this path as per your actual image

    for graph in columns:
        folder = f'{folderInitial}/{pais}/p{prioridad}/{graph}'
        path = getGoodFile(folder)
        graph_image_paths.append(folder+'/'+path)
        graph_text_paths.append(folder+'/'+'note.txt')
    return graph_image_paths, graph_text_paths


def calculateSpace(slideSizeX: float, imgInchSize: float, howMuchi: int, size: tuple = (800, 600)):
    spaceBetweenEquiSpace = round(
        (slideSizeX-(imgInchSize*howMuchi))/(howMuchi+1), 2)
    spaceBetweenBorderSpace = round((slideSizeX-(imgInchSize*howMuchi))/2, 2)
    space = None
    typeOfSpace = None
    if spaceBetweenEquiSpace < 0.5 and spaceBetweenBorderSpace < 0.5:
        imgInchSize = (slideSizeX-(0.3*2))/howMuchi
        space = 0.3
        typeOfSpace = 'border'
    elif spaceBetweenEquiSpace < 0.5:
        space = spaceBetweenBorderSpace
        typeOfSpace = 'border'
    else:
        space = spaceBetweenEquiSpace
        typeOfSpace = 'equi'
    heightOfImg = imgInchSize*(size[1]/size[0])
    widthOfImg = imgInchSize
    return space, typeOfSpace, widthOfImg, heightOfImg


def addGraphAndText(slide, graph_image_path: str, graph_note_path: str, Yposition: float, slideData: tuple, folder: str, size: tuple = (800, 600), slideSize: tuple = (14.5, 7.5)):
    slideSizeX, slideSizeY = slideSize
    i, howMuchi = slideData
    imgInchSize = 4.5
    space, typeOfSpace, imgInchSize, heightOfImg = calculateSpace(
        slideSizeX, imgInchSize, howMuchi, size)
    graph = Image.open(graph_image_path)
    graph = graph.resize(size)
    graph_path = f"""{presentationsFolder}/{
        folder}/supportImage/resized_graph_{i}_{howMuchi}.png"""
    graph.save(graph_path)
    positionX = None
    if typeOfSpace == 'equi':
        positionX = space+((i-1)*(imgInchSize+space))
    else:
        positionX = space+((i-1)*imgInchSize)

    slide.shapes.add_picture(graph_path, Inches(positionX), Inches(Yposition),
                             width=Inches(imgInchSize), height=Inches(heightOfImg))
    if howMuchi == 2:
        space = 0.5
    doGreatComment(graph_note_path, (positionX, Yposition+heightOfImg+0.3,
                   imgInchSize-0.03, slideSizeY-space-Yposition-heightOfImg), slide=slide)


def getPriorityText(priority: str):
    if priority == "0":
        priorityText = 'New Rs'
    if priority == "1":
        priorityText = 'Priority 1'
    elif priority == "2":
        priorityText = 'Priority 2'
    elif priority == "3":
        priorityText = 'Priority 3'
    elif priority == "4":
        priorityText = 'Priority 4'
    else:
        priorityText = 'Priority 5'
    return priorityText


def makePresentation(MAC: bool = True, prioridades: list = ['1', '2', '3', '4', '5',], bigFolder=plotsFolder, imagesPerSlide: int = 3):
    if MAC:
        paises = ['CO', 'PE', 'CR',]
    else:
        paises = ['MX',]
    createFolder(bigFolder)
    prs = Presentation()
    for pais in paises:
        for priority in prioridades:
            graph_image_paths, graph_text_paths = getImagesAndNotes(
                bigFolder, pais, priority)
            width, height = (14.5, 7.5)
            prs.slide_width = Inches(width)
            prs.slide_height = Inches(height)
            createFolder(f'{bigFolder}/supportImage')
            output_dir = f'{presentationsFolder}/{bigFolder}'
            for i in range(0, len(graph_image_paths)):
                if i % imagesPerSlide == 0:
                    slide_layout = prs.slide_layouts[6]  # Using a blank layout
                    slide = prs.slides.add_slide(slide_layout)
                    country_image_path = f'resources/{pais}.png'
                    didi = f'resources/didi.png'
                    slide.shapes.add_picture(country_image_path, Inches(
                        0.5), Inches(0.1), width=Inches(0.75), height=Inches(0.75))
                    slide.shapes.add_picture(didi, Inches(
                        13), Inches(0.01), width=Inches(1.5), height=Inches(0.84375))
                titleStart = (1.3, 0.1)
                priorityText = getPriorityText(priority)
                if i % imagesPerSlide == 0:
                    if i//imagesPerSlide == 0:
                        add_text(slide, f'PERFORMANCE', font_size=0.5,
                                 bold=True, color=(252, 76, 2), position=(titleStart[0], titleStart[1], width-titleStart[0]-2-0.5, 0.75),)
                    elif i//imagesPerSlide == 1:
                        add_text(slide, f'MIXED', font_size=0.5,
                                 bold=True, color=(252, 76, 2), position=(titleStart[0], titleStart[1], width-titleStart[0]-2-0.5, 0.75),)
                    else:
                        add_text(slide, f'SERVICE', font_size=0.5,
                                 bold=True, color=(252, 76, 2), position=(titleStart[0], titleStart[1], width-titleStart[0]-2-0.5, 0.75),)
                    actionTitleStart = (0.5, titleStart[1]+0.75)
                    add_text(slide, f'Action Title', font_size=0.5,
                             bold=True, color=(0, 0, 0), position=(actionTitleStart[0], actionTitleStart[1], width-0.5, 0.5), vertical=True)
                    subTitleStart = (0.5, actionTitleStart[1]+0.5)
                    add_text(slide, f'{priorityText} - ', font_size=0.25,
                             bold=False, color=(0, 0, 0), position=(subTitleStart[0], subTitleStart[1], width-0.5, 0.4), vertical=True)
                imgTop = subTitleStart[1]+0.8
                addGraphAndText(slide, graph_image_paths[i], graph_text_paths[i],
                                imgTop, (i % imagesPerSlide+1, min(imagesPerSlide, len(graph_image_paths)-imagesPerSlide*(i//imagesPerSlide))), bigFolder)
                if i % imagesPerSlide == 0 and i//imagesPerSlide < 2:
                    space, a, imgInchSize, heightOfImg = calculateSpace(width, 4.5, min(
                        imagesPerSlide, len(graph_image_paths)-imagesPerSlide*(i//imagesPerSlide)))
                    add_text(slide, f'Insights', font_size=0.2,
                             bold=False, color=(255, 255, 255), position=(space, imgTop+heightOfImg, imgInchSize-0.03, 0.3), vertical=True,
                             BG=(252, 76, 2))

            # Save the presentation
        print(f'Acabé con {pais} en {bigFolder}!')
    if MAC:
        add = 'MAC'
    else:
        add = 'MX'
    prs.save(f'{output_dir}/{add}_{bigFolder}.pptx')


def main():
    bigFolders = [plotsFolder,]
    for bigFolder in bigFolders:
        if both:
            makePresentation(MAC=True, bigFolder=bigFolder)
            makePresentation(MAC=False, bigFolder=bigFolder)
        else:
            makePresentation(MAC=MAC, bigFolder=bigFolder)


if __name__ == '__main__':
    main()
