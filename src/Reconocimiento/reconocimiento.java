package Reconocimiento;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;
import javax.imageio.ImageIO;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
/**
 *
 * @authors     José Castro - CI 24635097
*               José Alvarez - CI 25038805
 */

public class reconocimiento {

    public static String archivo = null;
    
    public static boolean aplicarSURF(String billete, String ruta) {

        String modelo = "images//billete_"+billete;
        String prueba = null;
        
        if (archivo == null)
            prueba = ruta;
        else
            prueba = archivo;
        System.out.println("*-----------------*");
        System.out.println(billete);

		//Cargando las imágenes
        Mat objectImage = Highgui.imread(modelo, Highgui.CV_LOAD_IMAGE_COLOR);
        Mat sceneImage = Highgui.imread(prueba, Highgui.CV_LOAD_IMAGE_COLOR);

        MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
		//Detectando puntos clave
        featureDetector.detect(objectImage, objectKeyPoints);
        KeyPoint[] keypoints = objectKeyPoints.toArray();

        MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        descriptorExtractor.compute(objectImage, objectKeyPoints, objectDescriptors);

        Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(), Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar newKeypointColor = new Scalar(255, 0, 0);

        Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage, newKeypointColor, 0);

        MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
        featureDetector.detect(sceneImage, sceneKeyPoints);
		//Calculando descriptores
        descriptorExtractor.compute(sceneImage, sceneKeyPoints, sceneDescriptors);

        Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2, Highgui.CV_LOAD_IMAGE_COLOR);
        Scalar matchestColor = new Scalar(0, 255, 0);
        List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		//Correspondiendo imágenes del objeto (imagen modelo) y la escena (imagen de prueba)
        descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors, matches, 2);

		//Calculando la lista de correspondencias
        LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

        float nndrRatio = 0.7f;

        for (int i = 0; i < matches.size(); i++) {
            MatOfDMatch matofDMatch = matches.get(i);
            DMatch[] dmatcharray = matofDMatch.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);

            }
        }

        System.out.println("Correspondencias = " + goodMatchesList.size());
        if (goodMatchesList.size() >= 45) {
            System.out.println("Objeto encontrado.");

            List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
            List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

            LinkedList<Point> objectPoints = new LinkedList<>();
            LinkedList<Point> scenePoints = new LinkedList<>();

            for (int i = 0; i < goodMatchesList.size(); i++) {
                objectPoints.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
                scenePoints.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
            }

            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);

            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);
            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);
            obj_corners.put(0, 0, new double[]{0, 0});
            obj_corners.put(1, 0, new double[]{objectImage.cols(), 0});
            obj_corners.put(2, 0, new double[]{objectImage.cols(), objectImage.rows()});
            obj_corners.put(3, 0, new double[]{0, objectImage.rows()});

	    	//Transformando esquinas del objeto (imagen modelo) a esquinas de la escena (imagen de prueba)
            Core.perspectiveTransform(obj_corners, scene_corners, homography);
            
            Mat img = Highgui.imread(prueba, Highgui.CV_LOAD_IMAGE_COLOR);

            List<MatOfPoint> border = new ArrayList<MatOfPoint>();
            border.add(new MatOfPoint(new Point(scene_corners.get(0, 0) ), new Point(scene_corners.get(1, 0) ),new Point(scene_corners.get(2, 0) ), new Point(scene_corners.get(3, 0) )));
            Core.fillPoly(img, border, new Scalar(0, 0, 0));
                      
            //Dibujando correspondencias en la imagen
            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);

            Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage, sceneKeyPoints, goodMatches, matchoutput, matchestColor, newKeypointColor, new MatOfByte(), 2);

            Highgui.imwrite("matchoutput_"+billete, matchoutput);
            Highgui.imwrite("img_"+billete, img);
            archivo = "img_"+billete;
            return true;
        } else {
            System.out.println("Objeto no encontrado.");
        }

        return false;
    }
    
}
