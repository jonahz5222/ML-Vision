//
//  ImageViewController.swift
//  ML-Vision
//
//  Created by DALE MUSSER on 12/12/17.
//  Updated 10/26/18 for Xcode 10.0
//  Copyright © 2017 Tech Innovator. All rights reserved.
//
// http://www.wolfib.com/Image-Recognition-Intro-Part-1/
// https://developer.apple.com/machine-learning/

import UIKit
import CoreML
import Vision

class ImageViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var modelSwitcher: UISegmentedControl!
    
    let imagePicker = UIImagePickerController()
    
    var googleModelText = ""
    var vcg16ModelText = ""
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        textView.text = ""
        activityIndicator.hidesWhenStopped = true
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    @IBAction func cameraSelected(_ sender: Any) {
        takePhotoWithCamera()
    }
    

    @IBAction func photoLibrarySelected(_ sender: Any) {
        pickPhotoFromLibrary()
    }
    
    @IBAction func switchModelText(_ sender: UISegmentedControl) {
        if sender.selectedSegmentIndex == 0 {
            textView.text = vcg16ModelText
        } else {
            textView.text = googleModelText
        }
    }
    
    func takePhotoWithCamera() {
        if (!UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.camera)) {
            let alertController = UIAlertController(title: "No Camera", message: "The device has no camera.", preferredStyle: .alert)
            let okAction = UIAlertAction(title: "OK", style: .default, handler: nil)
            alertController.addAction(okAction)
            present(alertController, animated: true, completion: nil)
        } else {
            imagePicker.allowsEditing = false
            imagePicker.sourceType = .camera
            present(imagePicker, animated: true, completion: nil)
        }
    }
    
    func pickPhotoFromLibrary() {
        imagePicker.allowsEditing = false
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true, completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let pickedImage = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            imageView.contentMode = .scaleAspectFit
            imageView.image = pickedImage
            textView.text = ""
            
            guard let ciImage = CIImage(image: pickedImage) else {
                displayString(string: "Unable to convert image to CIImage.");
                return
            }
            
            detectScene(image: ciImage)
        }
        
        dismiss(animated: true, completion: nil)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }
    
    func displayString(string: String) {
        textView.text = textView.text + string + "\n";
    }
    
    func detectScene(image: CIImage) {
        displayString(string: "detecting scene...")
        
        // Load the ML model through its generated class
        
         guard let googleModel = try? VNCoreMLModel(for: GoogLeNetPlaces().model) else {
             displayString(string: "Can't load ML model.")
             return
         }
 
        guard let vcg16Model = try? VNCoreMLModel(for: VGG16().model) else {
            displayString(string: "Can't load ML model.")
            return
        }
        
        // Create a Vision request with completion handler
        let googleRequest = VNCoreMLRequest(model: googleModel) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
//                self?.activityIndicator.stopAnimating()
                self?.googleModelText = ""
                for result in results {
                    if self?.modelSwitcher.selectedSegmentIndex == 1 {
                        self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                        self?.googleModelText += "\(Int(result.confidence * 100))% \(result.identifier)" + "\n"
                        self?.activityIndicator.stopAnimating()
                    } else {
                        self?.googleModelText += "\(Int(result.confidence * 100))% \(result.identifier)" + "\n"
                    }
                }
            }
        }
        
        let vcg16Request = VNCoreMLRequest(model: vcg16Model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation],
                let _ = results.first else {
                    self?.displayString(string: "Unexpected result type from VNCoreMLRequest")
                    return
            }
            
            // Update UI on main queue
            DispatchQueue.main.async { [weak self] in
                //self?.activityIndicator.stopAnimating()
                self?.vcg16ModelText = ""
                for result in results {
                    if self?.modelSwitcher.selectedSegmentIndex == 0 {
                        self?.displayString(string: "\(Int(result.confidence * 100))% \(result.identifier)")
                        self?.vcg16ModelText += "\(Int(result.confidence * 100))% \(result.identifier)" + "\n"
                        self?.activityIndicator.stopAnimating()
                    } else {
                        self?.vcg16ModelText += "\(Int(result.confidence * 100))% \(result.identifier)" + "\n"
                    }
                    
                }
            }
        }
        
        activityIndicator.startAnimating()
       
        
        // Run the Core ML GoogLeNetPlaces classifier on global dispatch queue
        let handler = VNImageRequestHandler(ciImage: image)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([googleRequest,vcg16Request])
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.displayString(string: error.localizedDescription)
                    self?.activityIndicator.stopAnimating()
                }
            }
        }
    }

}



