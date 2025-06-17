package com.ebookfrenzy.deyecyeyestesting

import android.app.Activity
import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.ebookfrenzy.deyecyeyestesting.ml.Yolo5s
import com.ebookfrenzy.deyecyeyestesting.ml.EfficientnetModel // <-- change this to match your model name
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var takePhoto: Button
    lateinit var choosePhoto: Button
    lateinit var predict: Button
    lateinit var resultText: TextView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView7)
        takePhoto = findViewById(R.id.button4)
        choosePhoto = findViewById(R.id.button5)
        predict = findViewById(R.id.button)
        resultText = findViewById(R.id.textView2)

        takePhoto.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, 2)
        }

        choosePhoto.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(Intent.createChooser(intent, "Select Image"), 1)
        }

        predict.setOnClickListener {
            if (!::bitmap.isInitialized) {
                Toast.makeText(this, "Please load an image first.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // === Preprocess image ===
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val byteBuffer = processedImage.buffer

            // === YOLO Detection ===
            val model = Yolo5s.newInstance(this)
            val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 416, 416, 3), DataType.FLOAT32)
            inputFeature.loadBuffer(byteBuffer)
            val outputs = model.process(inputFeature)
            model.close()

            val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray
            val detections = parseYOLOOutput(outputArray, bitmap.width, bitmap.height)

            if (detections.isEmpty()) {
                resultText.text = "No objects detected"
                return@setOnClickListener
            }

            val bestBox = detections.maxByOrNull { it.confidence }!!
            val cropped = cropBitmap(bitmap, bestBox.rect)

            // === Classification using EfficientNet ===
            val classifier = EfficientnetModel.newInstance(this) // replace with your model class
            val resizeProcessor = ImageProcessor.Builder()
                .add(ResizeOp(240, 240, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            val tensorInput = TensorImage(DataType.FLOAT32)
            tensorInput.load(cropped)
            val inputImg = resizeProcessor.process(tensorInput)

            val classifierInput = TensorBuffer.createFixedSize(intArrayOf(1, 240, 240, 3), DataType.FLOAT32)
            classifierInput.loadBuffer(inputImg.buffer)

            val classifierOutput = classifier.process(classifierInput)
            classifier.close()

            val outputProb = classifierOutput.outputFeature0AsTensorBuffer.floatArray
            val label = if (outputProb[0] > outputProb[1]) "Non-Fresh" else "Fresh"
            val confidence = outputProb.maxOrNull() ?: 0f

            // === Draw detection box ===
            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)
            val paint = Paint().apply {
                color = Color.RED
                style = Paint.Style.STROKE
                strokeWidth = 4f
            }
            canvas.drawRect(bestBox.rect, paint)

            imageView.setImageBitmap(mutableBitmap)
            resultText.text = "Detected Eye:\n${bestBox.rect}\n\nClassified: $label\nConfidence: ${"%.2f".format(confidence)}"
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1 && data?.data != null) {
                val uri = data.data
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                imageView.setImageBitmap(bitmap)
            } else if (requestCode == 2 && data?.extras?.get("data") != null) {
                bitmap = data.extras?.get("data") as Bitmap
                imageView.setImageBitmap(bitmap)
            }
        }
    }

    data class DetectionResult(val rect: RectF, val confidence: Float)

    private fun parseYOLOOutput(output: FloatArray, origW: Int, origH: Int): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()
        val numDetections = output.size / 6

        for (i in 0 until numDetections) {
            val offset = i * 6
            val x = output[offset]
            val y = output[offset + 1]
            val w = output[offset + 2]
            val h = output[offset + 3]
            val objConf = output[offset + 4]
            val classConf = output[offset + 5]
            val finalConf = objConf * classConf

            if (finalConf > 0.5f) {
                val left = (x - w / 2f) * origW
                val top = (y - h / 2f) * origH
                val right = (x + w / 2f) * origW
                val bottom = (y + h / 2f) * origH

                results.add(DetectionResult(RectF(left, top, right, bottom), finalConf))
            }
        }
        return results
    }

    private fun cropBitmap(bitmap: Bitmap, rect: RectF): Bitmap {
        val left = rect.left.toInt().coerceIn(0, bitmap.width - 1)
        val top = rect.top.toInt().coerceIn(0, bitmap.height - 1)
        val right = rect.right.toInt().coerceIn(left + 1, bitmap.width)
        val bottom = rect.bottom.toInt().coerceIn(top + 1, bitmap.height)
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
    }
}
