/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

import UIKit
import MetalKit
import CoreMedia
import paddle_mobile
import MetalPerformanceShaders

enum Platform {
  case GPU
}

let platformSupport: [(Platform, String)] = [(.GPU, "GPU")]

enum SupportModel: String{
  //  case mobilenet = "mobilenet"
  //  case mobilenet_ssd    = "mobilenetssd"
  case yolo               = "yolo"
  case mobilenet_combined = "mobilenet_combined"
  case super_resolution   = "superresoltion"
  case mobilenet          = "mobilenet"
  
  static func supportedModels() -> [SupportModel] {
    // .mobilenet,
    // .mobilenet_ssd,
    return [.super_resolution, .yolo, .mobilenet_combined, .mobilenet]
  }
}

let netSupport: [SupportModel : Net] = [
  .super_resolution : SuperResolutionNet.init(device: MetalHelper.shared.device),
  .yolo : YoloNet.init(device: MetalHelper.shared.device),
  .mobilenet_combined : MobileNetCombined.init(device: MetalHelper.shared.device),
  .mobilenet : MobileNet.init(device: MetalHelper.shared.device)]

class ViewController: UIViewController {
  @IBOutlet weak var resultTextView: UITextView!
  @IBOutlet weak var selectImageView: UIImageView!
  @IBOutlet weak var elapsedTimeLabel: UILabel!
  @IBOutlet weak var modelPickerView: UIPickerView!
  @IBOutlet weak var threadPickerView: UIPickerView!
  @IBOutlet weak var videoView: UIView!
  //  var videoCapture: VideoCapture!
  
  var selectImage: UIImage?
  var inputPointer: UnsafeMutablePointer<Float32>?
  var modelType: SupportModel = SupportModel.supportedModels()[0]
  var toPredictTexture: MTLTexture?
  
  var runner: Runner!
  var platform: Platform = .GPU
  var threadNum = 1
  
  @IBAction func loadAct(_ sender: Any) {
    runner = Runner.init(inNet: netSupport[modelType]!, commandQueue: MetalHelper.shared.queue)
    if platform == .GPU {
      if self.toPredictTexture == nil {
        runner.getTexture(image: selectImage!.cgImage!) { [weak self] (texture) in
          self?.toPredictTexture = texture
        }
      }
    } else {
      fatalError( " unsupport " )
    }
    
    if runner.load() {
      print(" load success ! ")
    } else {
      print(" load error ! ")
    }
  }
  
  @IBAction func selectImageAct(_ sender: Any) {
    let imagePicker = UIImagePickerController()
    imagePicker.sourceType = .camera
    imagePicker.delegate = self
    self.present(imagePicker, animated: true, completion: nil)
  }
  
  @IBAction func clearAct(_ sender: Any) {
    runner.clear()
  }
  
  @IBAction func predictAct(_ sender: Any) {
    let max = 1
    switch platform {
    case .GPU:
      guard let inTexture = toPredictTexture else {
        resultTextView.text = "请选择图片 ! "
        return
      }
      
      let startDate = Date.init()
      for i in 0..<max {
        self.runner.predict(texture: inTexture) { [weak self] (success, resultHolder)  in
          guard let sSelf = self else {
            fatalError()
          }
          if success {
            if i == max - 1 {
              let time = Date.init().timeIntervalSince(startDate)
              print(Array<Any>.floatArrWithBuffer(floatArrBuffer: resultHolder!.result!, count: resultHolder!.capacity).strideArray())
              DispatchQueue.main.async {
                //                print(resultHolder!.result![0])
                sSelf.resultTextView.text = sSelf.runner.net.resultStr(res: resultHolder!)
                sSelf.elapsedTimeLabel.text = "平均耗时: \(time/Double(max) * 1000.0) ms"
                
              }
            }
          }
          
          DispatchQueue.main.async {
            resultHolder?.releasePointer()
          }
        }
      }
    }
  }
  
  override func viewDidLoad() {
    super.viewDidLoad()
    
    modelPickerView.delegate = self
    modelPickerView.dataSource = self
    threadPickerView.delegate = self
    threadPickerView.dataSource = self
    if let image = UIImage.init(named: "test.jpg") {
      selectImage = image
      selectImageView.image = image
    } else {
      print("请添加测试图片")
    }
    
    
    //    if platform == .CPU {
    //      inputPointer = runner.preproccess(image: selectImage!.cgImage!)
    //    } else if platform == .GPU {
    //      runner.getTexture(image: selectImage!.cgImage!) {[weak self] (texture) in
    //        self?.toPredictTexture = texture
    //      }
    //    } else {
    //      fatalError( " unsupport " )
    //    }
    
    //    videoCapture = VideoCapture.init(device: MetalHelper.shared.device, orientation: .portrait, position: .back)
    //    videoCapture.fps = 30
    //    videoCapture.delegate = self
    //    videoCapture.setUp { (success) in
    //      DispatchQueue.main.async {
    //        if let preViewLayer = self.videoCapture.previewLayer {
    //          self.videoView.layer.addSublayer(preViewLayer)
    //          self.videoCapture.previewLayer?.frame = self.videoView.bounds
    //        }
    //        self.videoCapture.start()
    //      }
    //    }
    
  }
}

extension ViewController: UIPickerViewDataSource, UIPickerViewDelegate{
  func numberOfComponents(in pickerView: UIPickerView) -> Int {
    if pickerView == modelPickerView {
      return 1
    } else if pickerView == threadPickerView {
      return 1
    } else {
      fatalError()
    }
  }
  
  func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
    if pickerView == modelPickerView {
      return SupportModel.supportedModels().count
    } else if pickerView == threadPickerView {
      return platformSupport.count
    } else {
      fatalError()
    }
  }
  
  public func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
    if pickerView == modelPickerView {
      return SupportModel.supportedModels()[row].rawValue
    } else if pickerView == threadPickerView {
      return platformSupport[row].1
    } else {
      fatalError()
    }
  }
  
  public func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
    if pickerView == modelPickerView {
      self.modelType = SupportModel.supportedModels()[row]
    } else if pickerView == threadPickerView {
      platform = platformSupport[row].0
    } else {
      fatalError()
    }
  }
}

extension ViewController:  UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
    picker.dismiss(animated: true){[weak self] in
      guard let sSelf = self, let image =  info["UIImagePickerControllerOriginalImage"] as? UIImage else{
        fatalError("no image")
      }
      sSelf.selectImage = image
      sSelf.selectImageView.image = image
      sSelf.runner.getTexture(image: image.cgImage!, getTexture: { (texture) in
        sSelf.toPredictTexture = texture
      })
    }
  }
}

var bool1 = false
extension ViewController: VideoCaptureDelegate{
  func predictTexture(texture: MTLTexture){
    runner.scaleTexture(input: texture) { (scaledTexture) in
      self.runner.predict(texture: scaledTexture, completion: { (success, resultHolder) in
        //        print(resultHolder!.result![0])
        resultHolder?.releasePointer()
      })
    }
  }

}




