# bingo_reader
OpenCVを用いたビンゴの番号を画像から自動認識するプログラム

## ファイル構成と使い方
### 基本的な使い方
dataディレクトリに解析したい画像を入れておきます。  
bingo_reader.pyを起動すると順番に画像を解析していきます。  
解析結果はコンソール画面に出力されます。  
解析結果の画像はresultディレクトリに入ります。   
解析に失敗した画像はerrorディレクトリに入ります。  

### その他のファイル
experimentには実験的に使用したファイルなどが入っています。  
num_imgは数字判定をするときに使用する相関をとるための番号の「型」です。  

### 調節
画像を先に2240×3000くらいでトリミングしてしておくと調子がいい。  
画像の移り方や輝度に対して敏感なので、白黒化のための閾値調節が必要なときがある。

## サンプル
### 入力
<img src="https://github.com/Penguin8885/bingo_reader/blob/master/experiment/exam_data/3060.JPG" alt="サンプル画像" title="サンプル画像">

### 出力
青字が解析した番号です。赤字はインデックスです。
<img src="https://github.com/Penguin8885/bingo_reader/blob/master/experiment/process_sample/05_3060.JPG" alt="サンプル画像" title="サンプル画像">

