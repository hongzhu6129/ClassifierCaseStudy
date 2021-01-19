package Mushroom;

import MLTestFramework.*;

//Datamorphic Test Specification 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;
import javax.swing.JFileChooser;

import morphy.annotations.*;  
public class MushroomMorphisms extends MLTest<MushroomValue, Integer> {
	
	@TestSetContainer(
		inputTypeName = "MushroomValue",
		outputTypeName = "Integer")
	public TestPool<MushroomValue, Integer> tSuite = testSuite;
 
	@MakeSeed
	public void GenerateRandomTestCases(){	
		Random randomGenerator = new Random();
		TestCase<MushroomValue, Integer> tc;
		for (int i=0; i<100;i++){
			tc = new TestCase<MushroomValue, Integer>();;
			MushroomValue mr = new MushroomValue();
			for (int j=0; j<22; j++) {
				mr.value[j]=randomGenerator.nextInt(MushroomFeatures.size[j]);
			}
			tc.input = mr;
			tc.setFeature(TestDataFeature.original);
			tc.setType("Random");
			testSuite.addTestCase(tc);
		}
	}

	@MakeSeed
	public void readTestCasesFromCSVFile(){
		MushroomValue mr;
		TestCase<MushroomValue, Integer> tc;
		String directory = System.getProperty("user.dir");
		JFileChooser fileChooser = new JFileChooser(directory);
		String fileName;
		File file; 
		if (fileChooser.showOpenDialog(null)==JFileChooser.APPROVE_OPTION) {
			file = fileChooser.getSelectedFile();
			fileName = file.getName();
		}else {
			return; 
		};
		FileReader fileReader;
		try {
			fileReader = new FileReader(file);
		} catch (FileNotFoundException e1) {
			return;
		};
		BufferedReader reader = new BufferedReader(fileReader);
		String line;
		try {
			while ((line = reader.readLine()) != null) {
				tc = new TestCase<MushroomValue, Integer>();;
				mr = new MushroomValue();
				String[] numStrs = line.split(",");
				int numValue = 0; 
				switch (numStrs[MushroomFeatures.cap_shape+1]) {
					case "b": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "f": numValue = 2; break;
					case "k": numValue = 3; break; 
					case "s": numValue = 4; break;
					case "x": numValue = 5; break;
				}		
				mr.value[MushroomFeatures.cap_shape] = numValue; 
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.cap_surface+1]) {
					case "f": numValue = 0; break; 
					case "g": numValue = 1; break;
					case "s": numValue = 2; break;
					case "y": numValue = 3; break; 
				}		
				mr.value[MushroomFeatures.cap_surface] = numValue; 
		
				numValue = 0; 
				switch (numStrs[MushroomFeatures.cap_color+1]) {
					case "b": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "e": numValue = 2; break;
					case "g": numValue = 3; break; 
					case "n": numValue = 4; break;
					case "p": numValue = 5; break;
					case "r": numValue = 6; break;
					case "u": numValue = 7; break;
					case "w": numValue = 8; break;
					case "y": numValue = 9; break;
				}		
				mr.value[MushroomFeatures.cap_color] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.bruises+1]) {
					case "f": numValue = 0; break; 
					case "t": numValue = 1; break;
				}		
				mr.value[MushroomFeatures.bruises] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.odor+1]) {
					case "a": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "f": numValue = 2; break;
					case "l": numValue = 3; break; 
					case "m": numValue = 4; break;
					case "n": numValue = 5; break;
					case "p": numValue = 6; break;
					case "s": numValue = 7; break;
					case "y": numValue = 8; break;
				}		
				mr.value[MushroomFeatures.odor] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.gill_attachment+1]) {
					case "a": numValue = 0; break; 
					case "d": numValue = 1; break;
					case "f": numValue = 2; break;
					case "n": numValue = 3; break; 
				}		
				mr.value[MushroomFeatures.gill_attachment] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.gill_spacing+1]) {
					case "c": numValue = 0; break; 
					case "d": numValue = 1; break;
					case "w": numValue = 2; break;
				}		
				mr.value[MushroomFeatures.gill_spacing] = numValue;	
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.gill_size+1]) {
					case "b": numValue = 0; break; 
					case "n": numValue = 1; break;
				}		
				mr.value[MushroomFeatures.gill_size] = numValue;	
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.gill_color+1]) {
					case "b": numValue = 0; break; 
					case "e": numValue = 1; break;
					case "g": numValue = 2; break;
					case "h": numValue = 3; break; 
					case "k": numValue = 4; break;
					case "n": numValue = 5; break;
					case "o": numValue = 6; break;
					case "p": numValue = 7; break;
					case "r": numValue = 8; break;
					case "u": numValue = 9; break;
					case "w": numValue = 10; break;
					case "y": numValue = 11; break;
				}		
				mr.value[MushroomFeatures.gill_color] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_shape+1]) {
					case "e": numValue = 0; break; 
					case "t": numValue = 1; break;
				}		
				mr.value[MushroomFeatures.stalk_shape] = numValue;	
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_root+1]) {
				//	case "?": numValue = 0; break; 
					case "b": numValue = 0; break;
					case "c": numValue = 1; break;
					case "e": numValue = 2; break; 
					case "r": numValue = 3; break;
					case "u": numValue = 4; break;
					case "z": numValue = 5; break;
				}		
				mr.value[MushroomFeatures.stalk_root] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_surface_above_ring+1]) {
					case "f": numValue = 0; break; 
					case "k": numValue = 1; break;
					case "s": numValue = 2; break;
					case "y": numValue = 3; break; 
				}		
				mr.value[MushroomFeatures.stalk_surface_above_ring] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_surface_below_ring+1]) {
					case "f": numValue = 0; break; 
					case "k": numValue = 1; break;
					case "s": numValue = 2; break;
					case "y": numValue = 3; break; 
				}		
				mr.value[MushroomFeatures.stalk_surface_below_ring] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_color_above_ring+1]) {
					case "b": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "e": numValue = 2; break;
					case "g": numValue = 3; break; 
					case "n": numValue = 4; break;
					case "o": numValue = 5; break;
					case "p": numValue = 6; break;
					case "w": numValue = 7; break;
					case "y": numValue = 8; break;
				}		
				mr.value[MushroomFeatures.stalk_color_above_ring] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.stalk_color_below_ring+1]) {
					case "b": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "e": numValue = 2; break;
					case "g": numValue = 3; break; 
					case "n": numValue = 4; break;
					case "o": numValue = 5; break;
					case "p": numValue = 6; break;
					case "w": numValue = 7; break;
					case "y": numValue = 8; break;
				}		
				mr.value[MushroomFeatures.stalk_color_below_ring+1] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.veil_type]) {
					case "p": numValue = 0; break; 
					case "u": numValue = 1; break;
				}		
				mr.value[MushroomFeatures.veil_type] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.veil_color+1]) {
					case "n": numValue = 0; break; 
					case "o": numValue = 1; break;
					case "w": numValue = 2; break;
					case "y": numValue = 3; break; 
				}		
				mr.value[MushroomFeatures.veil_color] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.ring_number+1]) {
					case "n": numValue = 0; break; 
					case "o": numValue = 1; break;
					case "t": numValue = 2; break;
				}		
				mr.value[MushroomFeatures.ring_number] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.ring_type+1]) {
					case "c": numValue = 0; break; 
					case "e": numValue = 1; break;
					case "f": numValue = 2; break;
					case "l": numValue = 3; break; 
					case "n": numValue = 4; break;
					case "p": numValue = 5; break;
					case "s": numValue = 6; break;
					case "z": numValue = 7; break;
				}		
				mr.value[MushroomFeatures.ring_type] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.spore_print_color+1]) {
					case "b": numValue = 0; break; 
					case "h": numValue = 1; break;
					case "k": numValue = 2; break;
					case "n": numValue = 3; break; 
					case "o": numValue = 4; break;
					case "r": numValue = 5; break;
					case "u": numValue = 6; break;
					case "w": numValue = 7; break;
					case "y": numValue = 8; break;
				}		
				mr.value[MushroomFeatures.spore_print_color] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.population+1]) {
					case "a": numValue = 0; break; 
					case "c": numValue = 1; break;
					case "n": numValue = 2; break;
					case "s": numValue = 3; break; 
					case "v": numValue = 4; break;
					case "y": numValue = 5; break;
				}		
				mr.value[MushroomFeatures.population] = numValue;
				
				numValue = 0; 
				switch (numStrs[MushroomFeatures.habitat+1]) {
					case "d": numValue = 0; break; 
					case "g": numValue = 1; break;
					case "l": numValue = 2; break;
					case "m": numValue = 3; break; 
					case "p": numValue = 4; break;
					case "u": numValue = 5; break;
					case "w": numValue = 6; break;
				}		
				mr.value[MushroomFeatures.habitat] = numValue;

				tc.input = mr;
				switch (numStrs[0]) {
					case "e": tc.output = 0; break;
					case "p": tc.output = 1; break;
				}
				
				tc.setFeature(TestDataFeature.original);
				tc.setType("fromFile");
				testSuite.addTestCase(tc);
			}
		} catch (IOException e2) {
			e2.printStackTrace();
			System.out.println("Error in reading file.");
			return;
		}
		try {
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public TestCase<MushroomValue, Integer> FeatureUp (int feature, TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = new TestCase<MushroomValue, Integer>();
		MushroomValue mr = new MushroomValue();
		for (int i=0; i<22; i++) {
			mr.value[i] = seed.input.value[i];
		};
		if (seed.input.value[feature] < MushroomFeatures.size[feature]-2) {
			mr.value[feature]= seed.input.value[feature]+1;
		}
		mutant.input = mr;
		return mutant; 
	}
	
	public TestCase<MushroomValue, Integer> FeatureDown (int feature, TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = new TestCase<MushroomValue, Integer>();
		MushroomValue mr = new MushroomValue();
		for (int i=0; i<22; i++) {
			mr.value[i] = seed.input.value[i];
		};
		if (seed.input.value[feature] > 0) {
			mr.value[feature]= seed.input.value[feature]-1;
		}
		mutant.input = mr;
		return mutant; 
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_shape_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.cap_shape, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_shape_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.cap_shape, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_surface_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.cap_surface, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_surface_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.cap_surface, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_color_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.cap_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> cap_color_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.cap_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> bruises_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.bruises, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> bruises_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.bruises, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> odor_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.odor, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> odor_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.odor, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_attachment_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.gill_attachment, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_attachment_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.gill_attachment, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_spacing_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.gill_spacing, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_spacing_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.gill_spacing, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_size_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.gill_size, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_size_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.gill_size, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_color_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.gill_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> gill_color_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.gill_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_shape_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_shape, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_shape_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_shape, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_root_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_root, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_root_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_root, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_surface_above_ring_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_surface_above_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_surface_above_ring_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_surface_above_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_surface_below_ring_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_surface_below_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_surface_below_ring_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_surface_below_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_color_above_ring_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_color_above_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_color_above_ring_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_color_above_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_color_below_ring_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.stalk_color_below_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> stalk_color_below_ring_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.stalk_color_below_ring, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> veil_type_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.veil_type, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> veil_type_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.veil_type, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> veil_color_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.veil_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> veil_color_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.veil_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> ring_number_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.ring_number, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> ring_number_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.ring_number, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> ring_type_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.ring_type, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> ring_type_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.ring_type, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> spore_print_color_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.spore_print_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> spore_print_color_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.spore_print_color, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> population_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.population, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> population_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.population, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> habitat_Up (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureUp(MushroomFeatures.habitat, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> habitat_Down (TestCase<MushroomValue,Integer> seed){
		TestCase<MushroomValue, Integer> mutant = FeatureDown(MushroomFeatures.habitat, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<MushroomValue, Integer> mid (
				TestCase<MushroomValue,Integer> seed1, 
				TestCase<MushroomValue,Integer> seed2){
		TestCase<MushroomValue, Integer> mutant = new TestCase<MushroomValue, Integer>();
		MushroomValue mr = new MushroomValue();
		boolean toChange = true; 
		for (int i=0; i<22; i++) {
			mr.value[i] = seed1.input.value[i];
			if (seed1.input.value[i] != seed2.input.value[i]) {
				if (toChange) {
					mr.value[i] = seed2.input.value[i];
				}
				toChange = !toChange; 
			}
		}
		mutant.input = mr;
		return mutant ;
	}

	@Override
	public double distance(TestCase<MushroomValue, Integer> x, TestCase<MushroomValue, Integer> y) {
		int dist = 0;
		for (int i=0; i<=21; i++) {
			if (x.input.value[i]!=y.input.value[i]) {
				dist++;
			}
		}
		return dist;
	}
}
