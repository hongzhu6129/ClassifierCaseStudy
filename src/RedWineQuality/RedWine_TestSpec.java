package RedWineQuality;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.swing.filechooser.FileFilter;

import BankChurners.BankChurnersValue;
import MLTestFramework.ModelInvoker;
import MLTestFramework.PyFileFilter;
import morphy.annotations.*;  

public class RedWine_TestSpec extends RedWineMorphisms {
	@TestExecuter
	public Integer execute(RedWineFeatures x) {
		String arguements = x.fixedAcidity +"," 
				+ x.volatileAcidity +"," 
				+ x.citricAcid + ","
				+ x.residualSugar + ","
				+ x.chlorides +","
				+ x.freeSulfurDioxide +","
				+ x.totalSulfurDioxide + ","
				+ x.density + ","
				+ x.pH +","
				+ x.sulphates + ","
				+ x.alcohol;
		try { 
			String result = invoker.invokeModel(arguements);
			System.out.print("."); 
			int quality = Integer.valueOf(result);
			return quality;
		} catch (Exception e) {
			System.out.println("Failed to invoke Python script on parameters: " + arguements);
//			e.printStackTrace();
			return -1;
		}
	}
}
