# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, draw, transform
from skimage.filters import threshold_otsu
import os
import glob
import re



class DropletParse():
    """
    dropletの顕微鏡画像を解析するクラス
    """
    
    def __init__(self, file_name):
        self.img = io.imread(file_name)
        self.path = file_name

    def __repr__(self):
        return self.path

    def show_img(self):
        """
        ロードした画像ファイルを表示
        """
        plt.imshow(self.img, cmap='gray')

    
    def apply_rolling_ball(self):
        # rolling ballメソッド作成
        return None
    
    def apply_tophat(self):
        # top-hat処理メソッド作成

        return None


    def make_binary(self, algo='otsu', reverse=False, show=True):
        """
        読み込んだ画像の二値化を行い、その画像を表示・返す。
        Parameters
        ----------
        algo : {'otsu'}, optional
            使用アルゴリズムを指定。デフォルトはotsu
        reverse : bool, optional
            Trueにすると結果を反転する。デフォルトはFalse
        show : bool, optional
            Falseにすると画像を表示・デフォルトはTrue
        """
        if algo == 'otsu':
            thresh = threshold_otsu(self.img)

        if reverse == False:
            binary = self.img > thresh
        else:
            binary = self.img < thresh

        self.binary = binary.astype('uint16')
        
        if show == True:
            plt.imshow(self.binary, cmap='gray')

        return None
    
    def apply_mask(self, show=False):
        """
        元画像に対してマスク処理を行い、表示。
        Parameters
        ----------
        radius : bool, optional
            Trueにすると画像を表示。デフォルトはFalse
        """ 
        self.after_mask = self.img * self.binary
        if show == True:
            plt.imshow(self.after_mask, cmap='gray')

        return None

    def droplet_detection(self, radius=30, xmin=50, ymin=50, accum_rate=0.9, show=True):
        """
        ハフ変換でdropletを検出。近似円の半径、中心座標のリストを作成。検出した近似円を元画像に表示する。
        Parameters
        ----------
        radius : int, optional
            近似円の半径。デフォルトは30
        xmin, ymin : int, optional
            2円を判別する際の中心間の最小距離。デフォルトは50
        accum_rate : float, optional
            値を大きくすれば真円に近い近似円を返す。デフォルトは0.9、最大1.0。
        show : bool, optional
            Trueにすると画像を表示。デフォルトはTrue
        """
        hspaces = transform.hough_circle(self.binary, radius)
        #accum : どれだけ真円に近いか, cx/cy : 近似円の中心座標, rad : 近似円の半径
        accum, cx, cy, rad = transform.hough_circle_peaks(hspaces, [radius,], min_xdistance=xmin, min_ydistance=ymin)  

        if show == True:
            fig, ax = plt.subplots()

        self.cx = []
        self.cy = []
        self.accum = []
        self.rad = []

        for n, info in enumerate(zip(cx, cy, accum, rad)):
            #ハフ変換で求めた座標を中心に、半径radiusの円を描く
            if info[2] >= accum_rate:
                if show == True:
                    y, x = draw.circle_perimeter(info[1], info[0], radius)
                    ax.plot(x, y)
                    ax.text(info[0], info[1], n, size=5, horizontalalignment="center", verticalalignment="center")

                self.cx.append(info[0])
                self.cy.append(info[1])
                self.accum.append(info[2])
                self.rad.append(info[3])

        if show == True:
            ax.imshow(self.after_mask, cmap='gray')
            plt.show()

        return None

    def calculate_brightness(self):
        """
        ROI内部のdropletの輝度値を計算し、mean_listに格納。
        """
        #ハフ変換で求めた情報から各近似円でマスク画像を作り、元画像と重ね、その画像の輝度値の平均を求める。
        self.mean_list = []

        for y_, x_, rad_ in zip(self.cy, self.cx, self.rad):
            mask_ = np.zeros((2048, 2048), dtype='uint16')
            y, x = draw.circle(y_, x_, rad_)

            try:
                mask_[y, x] = 1
                roi_img = self.after_mask * mask_
                n_pixels = (mask_ == 1).sum()
                mean_value = np.sum(roi_img) / n_pixels
                self.mean_list.append(mean_value)

            except Exception:
                np.append(self.mean_list, 0)
        
        self.calculate_background()
        self.mean_list -= self.background
        
        return None
      
    def calculate_background(self):
        """
        backgroundの値を計算。
        """
        #メソッドの実行により値が書き変わるインスタンス変数を保持
        binary_temp = self.binary
        after_mask_temp = self.after_mask

        #反転二値化画像でマスク処理
        self.make_binary(reverse=True, show=False)
        self.apply_mask()

        #下位20%の値をbackgroundとする
        self.background = np.percentile(self.after_mask[self.after_mask != 0], 20)

        #インスタンス変数を戻す
        self.binary = binary_temp
        self.after_mask = after_mask_temp

        return None

    def draw_hist(self, bins=5):
        """
        dropletの輝度値の分布のヒストグラムを描く。
        Parameters
        ----------
        bins : int, optional
            ヒストグラムのビン。デフォルトは15
        """
        fig, ax = plt.subplots()

        ax.hist(self.mean_list, bins=bins)
        ax.set_title('distribution of brightness value')
        ax.set_xlabel('intensity')
        ax.set_ylabel('the number of droplets')

        fig.show()

    def make_result(self):
        """
        検出したdropletの情報のまとめを作成。
        """
        result_list = [self.mean_list, self.cx, self.cy, self.accum, self.rad]
        columns = ['brightness', 'x_coordinate', 'y_coordinate', 'accuracy', 'radius']
        
        result_dic = {}
        for list_, col in zip(result_list, columns):
            result_dic[col] = list_

        df_result = pd.DataFrame(result_dic)
        self.df_result = df_result[df_result['brightness'] > 0]

        return None

    def make_summary(self):
        """
        情報のsummary。(内訳はサンプルの数、平均、中央値、最大値、最小値、標準偏差)
        """
        count = len(self.mean_list)
        ave = np.mean(self.mean_list)
        med = np.median(self.mean_list)
        max_ = np.max(self.mean_list)
        min_ = np.min(self.mean_list)
        std = np.std(self.mean_list, ddof=1)

        summary_dic = {'count':[count], 'background':[self.background], 'average':[ave], 'median':[med], 'max':[max_], 'min':[min_], 'SD':[std]}
        self.df_summary = pd.DataFrame(summary_dic).T
        self.df_summary.columns = ['value']

        return None

    def export_csv(self):
        """
        resultとsummaryのcsvファイルを作成する。
        """
        self.make_result()
        self.make_summary()

        result_name = os.path.splitext(self.path)[0] + '_result.csv'
        summary_name = os.path.splitext(self.path)[0] + '_summary.csv'

        self.df_result.to_csv(result_name, index=False)
        self.df_summary.to_csv(summary_name)

    def excute_all(self, radius=30, xmin=50, ymin=50, accum_rate=0.9, algo='otsu', binary_show=False, droplet_show=True):
        """
        csvエクスポートまでのすべてのメソッドを実行。
        Parameters
        ----------
        binary_show : bool, optional
            Trueにするとbinary画像を表示する。デフォルトはFalse
        droplet_show : bool, optional
            Trueにするとdropletの検出画像を表示する。デフォルトはTrue
        others : optional
            各メソッドのものと同様
        """
        try:
            self.make_binary(algo=algo, show=binary_show)
            self.apply_mask()
            self.droplet_detection(radius=radius, xmin=xmin, ymin=ymin, accum_rate=accum_rate, show=droplet_show)
            self.calculate_brightness()
            self.export_csv()

            log = self.path + ' : Success'
            print(log)
      
        except Exception:
            log = self.path + ' : Error'
            print(log)


#テスト用
if __name__ == '__main__':
    print('テストするファイル名を入力(""は不要)')
    f = input()
    dp = DropletParse(f)
    dp.excute_all(radius=30)


