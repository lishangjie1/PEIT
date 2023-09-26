import os
from pydoc import describe
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from tqdm import tqdm
import pickle
import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
class ImgGenerator():
    def __init__(self, img_dir, fonts_dir, bg_path='./bg.pickle', min_size=20, max_size=30):
        
        self.fonts_dir = fonts_dir
        self.font_names = os.listdir(self.fonts_dir)
        self.min_size = min_size
        self.max_size = max_size
        self.fonts = self.load_font()
        self.img_dir = img_dir
        self.bg_path = bg_path
        if os.path.exists(bg_path):
            print("Loading existing backgrounds..")
            with open(bg_path, "rb") as f:
                self.bg = pickle.load(f)
        else:
            self.bg = self.load_backgrounds()
            print("Saving backgrounds..")
            with open(bg_path, "wb") as f:
                pickle.dump(self.bg, f)

    def load_font(self):
        fonts = {}
        for font_name in self.font_names:
            fonts[font_name] = {}
            for font_size in range(self.min_size, self.max_size):
                font_path = os.path.join(self.fonts_dir, font_name)
                fonts[font_name][font_size] = ImageFont.truetype(font_path, size=font_size, encoding="utf-8")
        return fonts

    def load_backgrounds(self):
        backgrounds = []
        description = os.path.join(self.img_dir, "image_descript.txt")
        print(f"Starting loading backgrounds")
        
        with open(description, "r") as f:
            content = f.readlines()
            for line in tqdm(content):
                background = {}
                items = line.strip().split("###")
                items[0] = items[0] if items[0].isprintable() else ''.join([x for x in items[0] if x.isprintable()])
                img_path = os.path.join(self.img_dir, items[0])
                if os.path.exists(img_path):
                    background["path"] = img_path
                    background["boxes"] = []
                else:
                    print(f"path of this background is not exist, skip it")
                    continue
                try:
                    for i in range(1,len(items)):
                        item = items[i]
                        splited = item.strip().split(",")
                        coord = eval(','.join(splited[:8]))
                        x_min, x_max, y_min, y_max = round(coord[0][0]), round(coord[1][0]), round(coord[0][1]), round(coord[2][1])
                        assert x_max > x_min
                        assert y_max > y_min
                        background["boxes"].append((x_min, x_max, y_min, y_max))
                except:
                    continue
                backgrounds.append(background)
        print(f"Successfully loading {len(backgrounds)} backgrounds")
        return backgrounds
    def region_proposal(self, height, width, write_height, write_width):
        regions = []
        x, y = 0,0
        while x < width and y < height:
            if x + write_width > width:
                x = 0
                y += write_height
                continue
            if y + write_height > height:
                break
            regions.append((x, x+write_width, y, y+write_height))
            x += write_width
        return regions
    def overlap(self, region1, region2):
        # IOU
        (x1, x2, y1, y2) = region1
        (a1, a2, b1, b2) = region2
        ax = max(x1, a1) 
        ay = max(y1, b1) 
        bx = min(x2, a2) 
        by = min(y2, b2) 
        
        w = bx - ax
        h = by - ay
        if w<=0 or h<=0:
            return False 
        return True
    def region_distance(self, img, region1, region2):
        subimg1 = img[region1[2]:region1[3],region1[0]:region1[1]]
        subimg2 = img[region2[2]:region2[3],region2[0]:region2[1]]
        return np.abs(subimg1 - subimg2).mean()
    

    def design_lines_layout(self, string, background, font):
        height, width, _ = background.shape
        cv2img = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        # cv2 to PIL
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        # design multi line layout
        lines = []
        write_height, write_width = 0, 0
        while len(string) > 0:
            left, right = 0, len(string)
            def binary_search(string, left, right):
                if left < right:
                    mid = (left + right) // 2 + 1
                    w, h = draw.textsize(string[:mid], font)
                    if w > width:
                        return binary_search(string, left, mid-1)
                    elif w < width:
                        return binary_search(string, mid, right)
                    else:
                        return mid
                else:
                    return right
            split_index = binary_search(string, left, right)
            sub_str, string = string[:split_index], string[split_index:]
            lines.append(sub_str.strip())
            w, h = draw.textsize(sub_str.strip(), font)
            write_height += h
            write_width = max(write_width, w)
        lines = '\n'.join(lines)
        return lines, write_height, write_width

    
    def generate_real_img(self, string, output_path):

        # random font
        font_name = self.font_names[np.random.randint(0, len(self.font_names))]
        # random size
        size = np.random.randint(self.min_size, self.max_size)
        font = self.fonts[font_name][size]

        # random background
        bg_id = np.random.randint(0, len(self.bg))
        bg = self.bg[bg_id]

        # load background img
        path = bg["path"]
        img = cv2.imread(path)
        height, width, _ = img.shape
        boxes = bg["boxes"]
        
        # cv2 to PIL
        cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        # design multi-line drawing
        lines, write_height, write_width = self.design_lines_layout(string, img, font)
        lines_num = len(lines.strip().split("\n"))
        while lines_num > 13: # shrink the font size
            if size > self.min_size:
                size = (self.min_size + size) // 2
                font = self.fonts[font_name][size]
                lines, write_height, write_width = self.design_lines_layout(string, img, font)
                lines_num = len(lines.strip().split("\n"))
            else:
                return False, size, lines_num
        
        # region proposal
        regions = self.region_proposal(height, width, write_height, write_width)
        # random choose a valid region from background
        order = list(range(len(regions)))
        np.random.shuffle(order)
        valid_region = None
        for idx in order:
            region = regions[idx]
            valid = True
            for box in boxes:
                if self.overlap(region, box):
                    valid = False
                    break
            if valid:
                valid_region = region
                break
        if valid_region is None:
            return False, size, 0
        
        # random set color according to region
        color = tuple([np.random.randint(0,256) for _ in range(3)])
        org = (valid_region[0],valid_region[2])
        lines_num = len(lines.strip().split("\n"))
        draw.multiline_text(org, lines.strip(),font=font,fill=color)

        drawed_img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
        output = drawed_img[org[1]:org[1]+write_height, org[0]:org[0]+write_width]
        cv2.imwrite(output_path, output, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return True, size, lines_num


if __name__ == "__main__":
    img_dir = "/tmp/ecoit" # randomly extract background from ecoit dataset
    fonts_dir = "/tmp/fonts"
    bg_path = "/tmp/bg.pickle"
    
    generator = ImgGenerator(img_dir, fonts_dir, bg_path)

    total_num = 1000000
    total_sentence = 0
    cnt = 0
    total_line = 0
    max_line = 0
    shard = 10
    images_dir = "/tmp/data/it_data/train_images"
    mt_data = "/tmp/data/mt_data/train"
    langs = ["zh", "en"] # write the text from the first language into picture
    fs = [open(mt_data+"."+lang, "r") for lang in langs]
    ws = [open(f"{images_dir}/text.{lang}", "w") for lang in langs]
    while True:
        total_sentence  += 1
        lines = [f.readline() for f in fs]
        if not lines[0]:
            break
        
        output_path = os.path.join(images_dir, f"{cnt}.jpg")
        write_down, font_size, lines_num = generator.generate_real_img(lines[0], output_path)
        if write_down:
            total_line += lines_num
            max_line = max(max_line, lines_num)
            cnt += 1
            for i in range(len(ws)):
                ws[i].write(lines[i].strip() + "\n")
            if cnt >= total_num:
                break

    print(f"total sentence: {total_sentence}")
    print(f"average line: {total_line / total_num}, max line:{max_line}")

            
    
    