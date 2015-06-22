package rewriteCSV;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;





import java.io.OutputStreamWriter;
import java.io.Writer;

import ucar.ma2.Array;
import ucar.ma2.Index;
import ucar.ma2.InvalidRangeException;
import ucar.nc2.NetcdfFile;
import ucar.nc2.Variable;

public class main {
//http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/tutorial/NetcdfFile.html
	
	public static void main(String[] args) throws InvalidRangeException {
		start();
	}

	public static void start()
	{
		String csvFile = "trainin_set.csv";
		String newFile = "improved_csv.csv";
		
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		try {
			try (Writer writer = new BufferedWriter(new OutputStreamWriter(
						new FileOutputStream(newFile), "utf-8"))) {
				
							br = new BufferedReader(new FileReader(csvFile));
							String headerLine = br.readLine();
							String[] newLineList = addExtraHeaders(headerLine.split(cvsSplitBy));
							writeLine(writer,newLineList,new double[0]);
							while ((line = br.readLine()) != null) 
							{
								String[] pump = line.split(cvsSplitBy);

								double[] tempList = insertTemperature(pump);
								writeLine(writer,pump,tempList);
							}
							
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private static String[] addExtraHeaders(String[] headerLine) {
		String[] newLine = new String[headerLine.length+17];
		for(int i =0; i < headerLine.length; i++){
			newLine[i] = headerLine[i];
		}
		newLine[headerLine.length] = "Month 1";
		newLine[headerLine.length+1] = "Month 2";
		newLine[headerLine.length+2] = "Month 3";
		newLine[headerLine.length+3] = "Month 4";
		newLine[headerLine.length+4] = "Month 5";
		newLine[headerLine.length+5] = "Month 6";
		newLine[headerLine.length+6] = "Month 7";
		newLine[headerLine.length+7] = "Month 8";
		newLine[headerLine.length+8] = "Month 9";
		newLine[headerLine.length+9] = "Month 10";
		newLine[headerLine.length+10] = "Month 11";
		newLine[headerLine.length+11] = "Month 12";
		newLine[headerLine.length+12] = "Winter average";
		newLine[headerLine.length+13] = "Spring average";
		newLine[headerLine.length+14] = "Summer average";
		newLine[headerLine.length+15] = "Autumn average";
		newLine[headerLine.length+16] = "Average";
		
		return newLine;
	}

	public static double[] insertTemperature(String[] lineElements)
	{
		double[] tempList = new double[12];
		double lon = Double.parseDouble(lineElements[6]);
		double lat = Double.parseDouble(lineElements[7]);
		double latMin = 0.0;
		double latMax = 0.0;
		double lonMin = 0.0;
		double lonMax = 0.0;
		//double lon = -62.0;
		//double lat = -26.0;
		if(lat != 0.0 && lat != -2.0E-8)
		{
			
			String filename = "absolute.nc";
			NetcdfFile ncfile = null;
			int latIndex = 0;
			int lonIndex = 0;
			  try {
			    ncfile = NetcdfFile.open(filename);
			    String latVarName = "lat"; 
			    Variable v = null;
			    
			     try {
			    	 v = ncfile.findVariable(latVarName);
			    	 Array latData = v.read();
			    	  int[] shape = latData.getShape();
			    	  Index index = latData.getIndex();

			    	  
			    	    for (int i=0; i<shape[0]; i++) 
			    	    {
			    	    	if(lat < latData.getDouble(index.set(i)))
			    	    	{
			    	    		latIndex = i;
			    	    		latMin = latData.getDouble(index.set(i+1));
			    	    		latMax = latData.getDouble(index.set(i));
			    	    	}
			    	    }

			    	  //NCdumpW.printArray(data, varName,System.out, null);
			    	 } catch (IOException ioe) {

			    	  }
			     
				     String lonVarName = "lon"; 
				     v = null;
			     try {
						v = ncfile.findVariable(lonVarName);
						Array lonData = v.read();
						int[] shape = lonData.getShape();
						Index index = lonData.getIndex();

			    	  
			    	    for (int i=0; i<shape[0]; i++) 
			    	    {
			    	    	double currentValue = lonData.getDouble(index.set(i));
			    	    	if(lon > currentValue)
			    	    	{
			    	    		lonIndex = i;
			    	    		lonMin = currentValue;
			    	    		lonMax = lonData.getDouble(index.set(i+1));
			    	    	}
			    	    }
			    	  //NCdumpW.printArray(data, varName,System.out, null);
			    	 } 
			     catch (IOException ioe) 
			     {
			    		 System.out.println("trying to read " + lonVarName +ioe);

		    	  }			     
				  try {
				    	 v = ncfile.findVariable("tem");
				    	 Array tempData = v.read();
							int[] shape = tempData.getShape();
							Index index = tempData.getIndex();
							double lonMaxDifference = (lonMax - lon)/5.0;
							double lonMinDifference = (lon - lonMin)/5.0;
							
							double latMaxDifference = (latMax - lat)/5.0;
							double latMinDifference = (lat - latMin)/5.0;

				    	    for (int i=0; i<shape[0]; i++) 
				    	    {
				    	    	double lonMinlatMinValue = tempData.getDouble(index.set(i,latIndex+1,lonIndex));
				    	    	double lonMaxlatMinValue = tempData.getDouble(index.set(i,latIndex+1,lonIndex+1));
				    	    	double lonMinlatMaxValue = tempData.getDouble(index.set(i,latIndex,lonIndex));
				    	    	double lonMaxlatMaxValue = tempData.getDouble(index.set(i,latIndex,lonIndex+1));
				    	    	
				    	    	double value = latMinDifference*(lonMinlatMinValue*lonMinDifference+lonMaxlatMinValue*lonMaxDifference)+latMaxDifference*(lonMinlatMaxValue*lonMinDifference+lonMaxlatMaxValue*lonMaxDifference);
				    	    	tempList[i] = (value/100);
				    	    }

				    	  //NCdumpW.printArray(data, varName,System.out, null);
				    	 } catch (IOException ioe) {
				    		 System.out.println("trying to read " + latVarName +ioe);

				    	  }
				  
				  
			  } catch (IOException ioe) {
				  System.out.println("trying to open " + filename + ioe);
			  } finally { 
			    if (null != ncfile) try {
			      ncfile.close();
			    } catch (IOException ioe) {
			    	System.out.println("trying to close " + filename + ioe);
			    }
			    
			  }
			  

			  
			  
			  
			  
		}
		else
		{
			
		}
		return tempList;
		
	}
	
	
	public static void writeLine(Writer writer, String[] lineElements, double[] tempList)
	{
		String line = "";
		double totalTemp = 0;
		for(int i = 0; i < lineElements.length; i++)
		{
			line += lineElements[i]+",";
		}
		for(int i = 0; i < tempList.length; i++)
		{
			line += String.valueOf(tempList[i])+",";
			totalTemp +=tempList[i];
		}
		if(tempList.length > 0)
		{
			line+=String.valueOf((tempList[0] + tempList[1] +tempList[11])/3)+",";
			line+=String.valueOf((tempList[3] + tempList[4] +tempList[2])/3)+",";
			line+=String.valueOf((tempList[6] + tempList[7] +tempList[5])/3)+",";
			line+=String.valueOf((tempList[9] + tempList[10] +tempList[8])/3)+",";
			line+=String.valueOf(totalTemp/12);
		}

		line += "\n";
		try {
			writer.write(line);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}