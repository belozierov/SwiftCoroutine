//
//  DispatchSourceTimer+extensions.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 06.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Dispatch

extension DispatchSourceTimer {
    
    @inlinable internal func start() {
        if #available(OSX 10.12, iOS 10.0, *) {
            activate()
        } else {
            resume()
        }
    }
    
}
