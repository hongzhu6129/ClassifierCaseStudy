package Mushroom;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import morphy.annotations.*;  

public class Mushroom_TestSpec extends MushroomMorphisms {
	
	@TestExecuter
	public Integer execute(MushroomValue x) {
		String arguements = "";
		for (int i=0; i<22; i++) {
			arguements = arguements + "," + x.value[i];
		}
		arguements = arguements.substring(1); 
		try { 
			String result = invoker.invokeModel(arguements);
			System.out.print("."); 
			int mushroomClass = Integer.valueOf(result);
			return mushroomClass;
		} catch (Exception e) {
			System.out.println("Failed to invoke Python script on parameters: " + arguements);
//			e.printStackTrace();
			return -1;
		}
	}
}
