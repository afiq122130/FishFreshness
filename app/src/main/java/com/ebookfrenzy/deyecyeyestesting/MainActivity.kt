package com.ebookfrenzy.deyecyeyestesting

import android.app.Activity
import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import android.net.Uri
import java.io.File
import android.os.Environment
import androidx.core.content.FileProvider
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface

import androidx.appcompat.app.AppCompatActivity
import com.ebookfrenzy.deyecyeyestesting.ml.Yolo5s
import com.ebookfrenzy.deyecyeyestesting.ml.EfficientnetModel
import com.ebookfrenzy.deyecyeyestesting.ml.ImprovisedValidFalseEfficientnetModel
import com.ebookfrenzy.deyecyeyestesting.ml.ImprovisedValidTrueEfficientnetModel
import com.ebookfrenzy.deyecyeyestesting.ml.Yolo5sImprovised
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var photoUri: Uri
    private lateinit var photoFile: File
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
            val photoIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)

            photoFile = File.createTempFile("IMG_", ".jpg", getExternalFilesDir(Environment.DIRECTORY_PICTURES))
            photoUri = FileProvider.getUriForFile(this, "$packageName.fileprovider", photoFile)

            photoIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
            startActivityForResult(photoIntent, 2)
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

            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(416, 416, ResizeOp.ResizeMethod.BILINEAR))
                .add(NormalizeOp(0f, 255f))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)
            val byteBuffer = processedImage.buffer

            //Test other model
            //val model = Yolo5s.newInstance(this)
            val model = Yolo5sImprovised.newInstance(this)

            val inputFeature = TensorBuffer.createFixedSize(intArrayOf(1, 416, 416, 3), DataType.FLOAT32)
            inputFeature.loadBuffer(byteBuffer)
            val outputs = model.process(inputFeature)
            model.close()

            val outputArray = outputs.outputFeature0AsTensorBuffer.floatArray
            val rawDetections = parseYOLOOutput(outputArray, bitmap.width, bitmap.height)
            val detections = applyNMS(rawDetections, iouThreshold = 0.4f)

            if (detections.isEmpty()) {
                resultText.text = "No Eye detected"
                return@setOnClickListener
            }

            //Try new model
            //val classifier = EfficientnetModel.newInstance(this)
            //val classifier = ImprovisedValidFalseEfficientnetModel.newInstance(this)
            val classifier = ImprovisedValidTrueEfficientnetModel.newInstance(this)


            val resizeProcessor = ImageProcessor.Builder()
                .add(ResizeOp(240, 240, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutableBitmap)

            detections.forEachIndexed { index, detection ->
                val cropped = cropBitmap(bitmap, detection.rect)
                val tensorInput = TensorImage(DataType.FLOAT32)
                tensorInput.load(cropped)
                val inputImg = resizeProcessor.process(tensorInput)

                val classifierInput = TensorBuffer.createFixedSize(intArrayOf(1, 240, 240, 3), DataType.FLOAT32)
                classifierInput.loadBuffer(inputImg.buffer)

                val classifierOutput = classifier.process(classifierInput)
                val outputProb = classifierOutput.outputFeature0AsTensorBuffer.floatArray

                val label = if (outputProb[0] > outputProb[1]) "Non-Fresh" else "Fresh"

                val paint = Paint().apply {
                    color = if (label == "Fresh") Color.GREEN else Color.RED
                    style = Paint.Style.STROKE
                    strokeWidth = 4f
                }

                canvas.drawRect(detection.rect, paint)
            }

            classifier.close()

            imageView.setImageBitmap(mutableBitmap)
            resultText.text = "\uD83D\uDFE9 Fresh   \uD83D\uDFE5 Non-Fresh" // 🟩 🟥
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == 1 && data?.data != null) {
                val uri = data.data
                bitmap = MediaStore.Images.Media.getBitmap(contentResolver, uri)
                imageView.setImageBitmap(bitmap)
            } else if (requestCode == 2 && resultCode == Activity.RESULT_OK) {
                val fullBitmap = BitmapFactory.decodeFile(photoFile.absolutePath)

                val exif = ExifInterface(photoFile.absolutePath)
                val orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
                )

                bitmap = when (orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90 -> rotateBitmap(fullBitmap, 90f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> rotateBitmap(fullBitmap, 180f)
                    ExifInterface.ORIENTATION_ROTATE_270 -> rotateBitmap(fullBitmap, 270f)
                    else -> fullBitmap
                }

                imageView.setImageBitmap(bitmap)
            }
        }
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
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

    private fun applyNMS(detections: List<DetectionResult>, iouThreshold: Float = 0.5f): List<DetectionResult> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val finalDetections = mutableListOf<DetectionResult>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            finalDetections.add(best)

            val toRemove = mutableListOf<DetectionResult>()
            for (other in sorted) {
                val iou = calculateIoU(best.rect, other.rect)
                if (iou > iouThreshold) {
                    toRemove.add(other)
                }
            }
            sorted.removeAll(toRemove)
        }
        return finalDetections
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val intersectionLeft = maxOf(a.left, b.left)
        val intersectionTop = maxOf(a.top, b.top)
        val intersectionRight = minOf(a.right, b.right)
        val intersectionBottom = minOf(a.bottom, b.bottom)

        val intersectionArea = maxOf(0f, intersectionRight - intersectionLeft) * maxOf(0f, intersectionBottom - intersectionTop)
        val aArea = (a.right - a.left) * (a.bottom - a.top)
        val bArea = (b.right - b.left) * (b.bottom - b.top)

        val unionArea = aArea + bArea - intersectionArea
        return if (unionArea == 0f) 0f else intersectionArea / unionArea
    }
}
