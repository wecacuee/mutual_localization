.PHONY: all data/two-camera/20120903/ data/tiledexperiment_results/ data/bnwmarker20120831/ ros/mutloc_ros/nodes/ ros/mutloc_ros/launch ros/mutloc_ros/src/ data/bnwmarker20120831/ 



all: src/mutloc/test/test_pose_computation.py scripts/artk_vs_mutloc_on_tiles.py scripts/noise_vs_error_on_blender.py scripts/print_target_localization.py doc/

src/mutloc/test/test_pose_computation.py : src/mutloc/test/.dirstamp src/mutloc/core.py src/mutloc/scalefactors.py src/mutloc/utils.py lib/transformations.py src/mutloc/test/__init__.py src/mutloc/__init__.py src/mutloc/config.py src/mutloc/findmarkers.py src/mutloc/log.py src/mutloc/absor.py lib/memoize.py
	cp ../$@ $@ 

lib/transformations.py: lib/.dirstamp
	cp ../lib/transformations.* ../lib/transformations.c lib/; \
	cd lib/; \
	python setup.py build

scripts/artk_vs_mutloc_on_tiles.py: scripts/.dirstamp data/tiledexperiment_results/ src/mutloc/filewriters.py scripts/compareartoolkit.py src/mutloc/mayaviutils.py lib/alignpointcloud.py media/.dirstamp ros/mutloc_ros/nodes/ ros/mutloc_ros/launch ros/mutloc_ros/src/
	cp ../$@ $@

doc/: doc/.dirstamp doc/source/.dirstamp
	cp ../$@/Makefile $@/
	rsync -ruE ../$@/source/ $@/source/

ros/mutloc_ros/:ros/mutloc_ros/.dirstamp
	rsync -uE ../$@/CMakeLists.txt ../$@/manifest.xml ../$@/mainpage.dox $@

ros/mutloc_ros/nodes/: ros/mutloc_ros/nodes/.dirstamp ros/mutloc_ros/
	rsync -ruE ../$@ $@

ros/mutloc_ros/launch/: ros/mutloc_ros/launch/.dirstamp ros/mutloc_ros/
	rsync -ruE ../$@ $@

ros/mutloc_ros/src: ros/mutloc_ros/src/.dirstamp ros/mutloc_ros/
	rsync -ruE ../$@ $@

scripts/noise_vs_error_on_blender.py: data/two-camera/20120903/ src/mutloc/localizer.py src/mutloc/surfmatcher.py
	cp ../$@ $@

scripts/print_target_localization.py: data/bnwmarker20120831/ lib/ransac.py src/mutloc/usbcam.py src/mutloc/mark_images.py src/mutloc/undistortimgs.py
	cp ../$@ $@

data/bnwmarker20120831/: data/bnwmarker20120831/.dirstamp
	rsync -ruE ../$@ $@

data/two-camera/20120903/: data/two-camera/20120903/.dirstamp
	rsync -ruE ../$@ $@

data/tiledexperiment_results/: data/tiledexperiment_results/.dirstamp
	rsync -ruE ../$@ $@

%.py:
	cp ../$@ $@

%/.dirstamp:
	mkdir -p $(dir $@)
	touch $@
