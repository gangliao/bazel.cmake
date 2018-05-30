# Copyright (c) 2016 Gang Liao <gangliao@umd.edu> All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  
  if(libs_deps)
    list(REMOVE_DUPLICATES libs_deps)
  endif()

  if(APPLE) # Use OSX's libtool to merge archives
    # To produce a library we need at least one source file,
    # which is created by add_custom_command below. It
    # also can help to track dependencies.
    set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

    # Make the generated dummy source file dependeds on all static input
    # libraries. If they are changed, the dummy file is also touched
    # which causes the desired effect (relink).
    add_custom_command(OUTPUT ${dummyfile}
      COMMAND ${CMAKE_COMMAND} -E touch ${dummyfile}
      DEPENDS ${libs})

    # Generate dummy staic library
    file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
    add_library(${TARGET_NAME} STATIC ${dummyfile})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the filenames of the merged libraries 
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND rm -rf "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
      COMMAND /usr/bin/libtool -static -o "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles})
  else() # general UNIX: use "ar" to extract objects and re-add to a common lib
    foreach(lib ${libs})
      set(objlistfile ${lib}.objlist) # list of objects in the input library
      set(objdir ${lib}.objdir)

      add_custom_command(OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
        DEPENDS ${lib})

      add_custom_command(OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ../${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      # Empty dummy source file that goes into merged library		
      set(mergebase ${lib}.mergebase.c)		
      add_custom_command(OUTPUT ${mergebase}		
        COMMAND ${CMAKE_COMMAND} -E touch ${mergebase}		
        DEPENDS ${objlistfile})		

      list(APPEND mergebases "${mergebase}")
    endforeach()

    add_library(${TARGET_NAME} STATIC ${mergebases})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    # Get the file name of the generated library
    set(outlibfile "$<TARGET_FILE:${TARGET_NAME}>")

    foreach(lib ${libs})
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_AR} cr ${outlibfile} *.o
        COMMAND ${CMAKE_RANLIB} ${outlibfile}
        WORKING_DIRECTORY ${lib}.objdir)
    endforeach()
  endif()
endfunction(merge_static_libs)
