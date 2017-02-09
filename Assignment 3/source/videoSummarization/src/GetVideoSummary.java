
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import javax.servlet.RequestDispatcher;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
import org.openimaj.math.model.fit.RANSAC;
import org.openimaj.video.Video;
import org.openimaj.video.xuggle.XuggleVideo;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.typography.hershey.HersheyFont;

import clarifai2.api.ClarifaiBuilder;
import clarifai2.api.ClarifaiClient;
import clarifai2.api.ClarifaiResponse;
import clarifai2.dto.input.ClarifaiInput;
import clarifai2.dto.input.image.ClarifaiImage;
import clarifai2.dto.model.output.ClarifaiOutput;
import clarifai2.dto.prediction.Concept;
import okhttp3.OkHttpClient;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;


@WebServlet("/getVideoSummary")
public class GetVideoSummary extends HttpServlet{
	
	private static final long serialVersionUID = 1L;

	static Video<MBFImage> video;
    static List<MBFImage> imageList = new ArrayList<MBFImage>();
    static List<Long> timeStamp = new ArrayList<Long>();
    static List<Double> mainPoints = new ArrayList<Double>();
	/**
	 * @see HttpServlet#doGet(HttpServletRequest request, HttpServletResponse
	 *      response)
	 */	
	protected void doGet(HttpServletRequest request, HttpServletResponse response)throws ServletException, IOException 
	{
		
		System.out.println("Servlet input file-----------" + getServletContext().getRealPath("/"));		
		
		String path = request.getParameter("path");
		path="E:/big data analytics/Lab Assignment 3/Image-Annotation/Clarifai/input/sample.mkv";
		System.out.println("Path-----------"+path);
		//Frames(path);
       // MainFrames();	            
        
        final ClarifaiClient client = new ClarifaiBuilder("LlL6lZ3SFaKvLHvGON2PZoFquuN11_GRVXiQgVBn", "D-PPdr_skEpWP5HthjgljF5x2g3QFd_EZmVrLnJX")
                .client(new OkHttpClient()) // OPTIONAL. Allows customization of OkHttp by the user
                .buildSync(); // or use .build() to get a Future<ClarifaiClient>
        client.getToken();

        File file = new File("E:/big data analytics/Lab Assignment 3/Image-Annotation/Clarifai/output/mainframes");
        File[] files = file.listFiles();
        List<String> finalData =new ArrayList<String>();
        HashMap<String,Float> map=new HashMap<>();
        for (int i=0; i<files.length;i++){
            ClarifaiResponse response1 = client.getDefaultModels().generalModel().predict()
                    .withInputs(
                            ClarifaiInput.forImage(ClarifaiImage.of(files[i]))
                    )
                    .executeSync();
            List<ClarifaiOutput<Concept>> predictions = (List<ClarifaiOutput<Concept>>) response1.get();
            MBFImage image = ImageUtilities.readMBF(files[i]);
            int x = image.getWidth();
            int y = image.getHeight();

            System.out.println("*************" + files[i] + "***********");
            List<Concept> data = predictions.get(0).data();
            
            for (int j = 0; j < data.size(); j++) {
                System.out.println(data.get(j).name() + " - " + data.get(j).value());
                //image.drawText(data.get(j).name(), (int)Math.floor(Math.random()*x), (int) Math.floor(Math.random()*y), HersheyFont.ASTROLOGY, 20, RGBColour.RED);
                 
                /*if(map.containsKey(data.get(j).name()))
    			{
    				
    				map.put(data.get(j).name(),map.get(data.get(j).name()).floatValue()+data.get(j).value());
    			}
    			else	
    			{  
    				map.put(data.get(j).name(),data.get(j).value());
    			}*/
                
                finalData.add(data.get(j).name());
            }
            //DisplayUtilities.displayName(image, "image" + i);
        }   
        
        //Collections.sort(finalData);
       // System.out.println("final Summary is"+Arrays.toString(map.toArray()));        	
		
        request.setAttribute("summaryList", finalData);
        
	    RequestDispatcher rd = request.getRequestDispatcher("videosummary.jsp");
	    rd.forward(request, response);
	}
	
	
	 public static void Frames(String path){
	        video = new XuggleVideo(new File(path));
//	        VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
	        int j=0;
	        for (MBFImage mbfImage : video) {
	            BufferedImage bufferedFrame = ImageUtilities.createBufferedImageForDisplay(mbfImage);
	            j++;
	            System.out.println("jindex is"+j);
	            String name = "E:/big data analytics/Lab Assignment 3/Image-Annotation/Clarifai/output/frames/new" + j + ".jpg";
	           
	            File outputFile = new File(name);

	            try {

	                ImageIO.write(bufferedFrame, "jpg", outputFile);

	            } catch (IOException e) {
	                e.printStackTrace();
	            }
	            MBFImage b = mbfImage.clone();
	            imageList.add(b);
	            timeStamp.add(video.getTimeStamp());
	        }
	    }

	    public static void MainFrames(){
	        for (int i=0; i<imageList.size() - 1; i++)
	        {
	            MBFImage image1 = imageList.get(i);
	            MBFImage image2 = imageList.get(i+1);
	            DoGSIFTEngine engine = new DoGSIFTEngine();
	            LocalFeatureList<Keypoint> queryKeypoints = engine.findFeatures(image1.flatten());
	            LocalFeatureList<Keypoint> targetKeypoints = engine.findFeatures(image2.flatten());
	            RobustAffineTransformEstimator modelFitter = new RobustAffineTransformEstimator(5.0, 1500,
	                    new RANSAC.PercentageInliersStoppingCondition(0.5));
	            LocalFeatureMatcher<Keypoint> matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
	                    new FastBasicKeypointMatcher<Keypoint>(8), modelFitter);
	            matcher.setModelFeatures(queryKeypoints);
	            matcher.findMatches(targetKeypoints);
	            double size = matcher.getMatches().size();
	            mainPoints.add(size);
	            System.out.println(size);
	        }
	        Double max = Collections.max(mainPoints);
	        for(int i=0; i<mainPoints.size(); i++){
	            if(((mainPoints.get(i))/max < 0.01) || i==0){
	                Double name1 = mainPoints.get(i)/max;
	                BufferedImage bufferedFrame = ImageUtilities.createBufferedImageForDisplay(imageList.get(i+1));
	                String name = "E:/big data analytics/Lab Assignment 3/Image-Annotation/Clarifai/output/mainframes/" + i + "_" + name1.toString() + ".jpg";
	                File outputFile = new File(name);
	                try {
	                    ImageIO.write(bufferedFrame, "jpg", outputFile);
	                } catch (IOException e) {
	                    e.printStackTrace();
	                }
	            }
	        }
	    }
	
	/**
	 * @see HttpServlet#doPost(HttpServletRequest request, HttpServletResponse
	 *      response)
	 */
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// TODO Auto-generated method stub
		doGet(request, response);
	}

}
