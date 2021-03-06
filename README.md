# 白雪姫の古文書研究ノート

<img src="https://user-images.githubusercontent.com/7970619/57186632-cae12c00-6f1d-11e9-9ccd-1a136344ce8a.jpg" align="right" width="240px">私の研究成果をまとめた「Pythonで機械学習入門-深層学習から敵対的生成ネットワークまで-」を読んでくれた読者の皆さんどうもありがとう。
おそらくここの場所を見つけられたということは、知識の泉を介してみんなも魔法の鏡を使っているということだよね。
大事な記録は魔法の鏡に付属していた星型の記録装置を使って残しておいたから、私が実際に書いた魔法の言葉をここに公開しておくね。

その魔法の言葉を、みんなの魔法の鏡に書き込んでも良いし、みんなが利用できる共同作業広場Colaboratoryというのもあって、そこで試しても良いように準備しておいたよ。

もしも魔法の言葉が正しく動作しないことがあった場合は [issues](https://github.com/mohzeki222/ohm_princess/issues) で教えてね。

![白雪姫１章_イラスト08](https://user-images.githubusercontent.com/7970619/57186627-c157c400-6f1d-11e9-8a43-debcab494b56.jpg)

## 誤植の発見
ごめんなさい！ちょっと間違えて書いてあったところ発見したよ。

初版一刷・Page 74
```
tdata = Iris.data.astype(np.int32)
```
は
```
tdata = Iris.target.astype(np.int32)
```
でした。

初版一刷・Page 307
```
data = data.astype(np.int8)
```
は
```
data = data.astype(np.uint8)
```
でした。unit8というのはプラスとかマイナスとか符号がない整数値を扱うという意味だよ。

Page 313
```
import PIL.image as im
```
は
```
import PIL.Image as im
```
でした。ごめんなさい！

Page344にohm.plot_resultという魔法を書いちゃったけど、
これはplot_result2の魔法を作り変えたもので、記録を残しておくのを忘れていたみたい。ごめんなさい。
自作魔法princess.pyで
```
def plot_result(result,title,xlabel,ylabel,ymin=0.0, ymax=1.0):
  Tall = len(result)
  plt.figure(figsize=(8,6))
  plt.plot(range(Tall), result1)
  plt.title(title) 
  plt.xlabel(xlabel)  
  plt.ylabel(ylabel)   
  plt.xlim([0,Tall])   
  plt.ylim([ymin,ymax])   
  plt.show()
```
というものを追加しましょう！

またはPage344については、
```
ohm.plot_result(result[0],"loss_ function of gen", "step","loss function",0.0,0.6)
```
を
```
ohm.plot_result2(result[0],result[0],"loss_ function of gen", "step","loss function",0.0,0.6) 
```
としても良いよ。

Page382にも同じようなところがあって、
```
ohm.plot_result(resultA[0],"loss_ function A to B of gen in training","step","loss function”,0.0,0.6)
```
は
```
ohm.plot_result2(resultA[0],resultA[0],"loss_ function A to B of gen in training","step","loss function”,0.0,0.6)
```
にして、
```
ohm.plot_result(resultB[0],"loss_ function B to A of gen in training","step","loss function",0.0,0.6)
```
の代わりに
```
ohm.plot_result2(resultB[0],resultB[0],"loss_ function B to A of gen in training","step","loss function",0.0,0.6)
```
とするのでも良いよ！

Page343で
```
ohm.temp_image(train_iter.epoch, foldername + "/test", data[0],cuda.to_gpu(ztrain),ztest,gen,dis)
```
とありますが、
```
ohm.temp_image(train_iter.epoch, foldername + "/test",data[0],ztest,gen,dis)
```
の間違いでした。

## 白雪姫の書いた魔法の言葉

* nbviewer: [nbviewer](https://nbviewer.jupyter.org)でレンダリングされたものが表示されるよ
* Open in Colab: [Google Colaboratory](https://colab.research.google.com/)上で実行できるよ

![お茶会](https://user-images.githubusercontent.com/7970619/57186613-6b831c00-6f1d-11e9-9cf7-5ad10a6552de.jpg)

章|タイトル|nbviewer|Open in Colab|自作魔法集利用|GPU利用|
-----|--------|--------|-------------|---------|---------
1|Chapter1.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter1.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter1.ipynb)|無|無
2|Chapter2.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter2.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter2.ipynb)|無|無
3|Chapter3.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter3.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter3.ipynb)|無|無
4|Chapter4-MNIST.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter4-MNIST.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter4-MNIST.ipynb)|無|無
　|Chapter4-fashion_mnist.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter4-fashion_mnist.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter4-fashion_mnist.ipynb)|有|有
5|Chapter5.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5.ipynb)|有|有
　|Chapter5-time.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5-time.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5-time.ipynb)|有|有
　|Chapter5-stock.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5-stock.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter5-stock.ipynb)|有|有
6|Chapter6-CIFAR10.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter6-CIFAR10.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter6-CIFAR10.ipynb)|有|有
　|Chapter6-ResBlock.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter6-ResBlock.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter6-ResBlock.ipynb)|有|有
7|Chapter7-princess_princess.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter7-princess_princess.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter7-princess_princess.ipynb)|有|有
　|Chapter7-GAN.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Chapter7-GAN.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Chapter7-GAN.ipynb)|有|有
付録|Appendix-CycleGAN.ipynb |[![nbviewer](https://camo.githubusercontent.com/bfeb5472ee3df9b7c63ea3b260dc0c679be90b97/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f72656e6465722d6e627669657765722d6f72616e67652e7376673f636f6c6f72423d66333736323626636f6c6f72413d346434643464)](https://nbviewer.jupyter.org/github/mohzeki222/ohm_princess/blob/master/notes/Appendix-CycleGAN.ipynb)|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohzeki222/ohm_princess/blob/master/notes/Appendix-CycleGAN.ipynb)|有|有

![こびと](https://user-images.githubusercontent.com/7970619/57187330-b0ac4b80-6f27-11e9-882a-f3b014b6a16d.jpg)

Google Colaboratoryで実行する場合は、
４章のChapter4-fashion_mnist.ipynbから自作魔法集「[princess.py](https://github.com/mohzeki222/ohm_princess/blob/master/notes/princess.py)」を利用するから、
最初に
```
from google.colab import files
uploaded = files.upload()
```
というColaboratoryのための魔法の言葉も追加しておいたよ。
これを実行するとファイルをColaboratoryへアップロードして利用することができるから自作魔法集を追加して試しに動かしてみてね。
あとGPUを利用するときは、「ランタイム」から「ランタイムのタイプを変更」、「ハードウェアアクセラレータ」で「GPU」を選択してから始めてね。
もしも今使っている魔法の鏡にGPUが潜んでいないようだったら、この機能を使って試してみると良いよ！

![GPU](https://user-images.githubusercontent.com/7970619/57194571-6062d700-6f83-11e9-9d31-a5c1422af698.png)

７章ではお妃様の画像と私の画像コレクションのダウンロード用のモジュール「download_figs()」を利用します。
これは日記帳には書いていないけど、「[princess.py](https://github.com/mohzeki222/ohm_princess/blob/master/notes/princess.py)」に追加しておいたよ。
```
!mkdir princess_fig
!mkdir white_fig
ohm.download_figs()
```
を実行すると画像コレクションがダウンロードされて利用できます。
勝手に他の目的で利用したらダメだよ。魔法の鏡に記録が残っているんだからね。

## 魔法の鏡の活躍ぶりの記録

お妃様のところに置かれた魔法の鏡はお城にすっかり馴染んで、この国を守っている様子。
その様子を見たかったら、王宮の図書館から黙って借りないで、知識の泉でここに行くと良いよ。

[「機械学習入門-深層学習からボルツマン機械学習まで-」](https://www.ohmsha.co.jp/book/9784274219986/)

[「ベイズ推定入門-モデル選択からベイズ的最適化まで-」](https://www.ohmsha.co.jp/book/9784274221392/)

<img src="https://user-images.githubusercontent.com/7970619/57194621-e0893c80-6f83-11e9-8480-25f3ce851c3d.jpg" align="right" width="240px">ふふふ。実は魔法の鏡は同期していて、お妃様の動きもキャッチしているのよ。国の発展のお役に立てて本当に良かった。
もちろん私と魔法の鏡の出会いの記録が書かれた本も知識の泉から見つかるよ。

[「Pythonで機械学習入門-深層学習から敵対的生成ネットワークまで-」](https://www.ohmsha.co.jp/book/9784274222863/)

![白雪姫３章_イラスト04](https://user-images.githubusercontent.com/7970619/57194602-a61f9f80-6f83-11e9-992c-d4bbfa51e054.jpg)

