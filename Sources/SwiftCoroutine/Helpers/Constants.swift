//
//  Constants.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Darwin

extension Int {
    
    internal static let pageSize = sysconf(_SC_PAGESIZE)
    internal static let processorsNumber = sysconf(_SC_NPROCESSORS_ONLN)
    internal static let environmentSize = MemoryLayout<jmp_buf>.size
    
}
