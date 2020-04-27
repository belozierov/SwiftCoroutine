//
//  CoroutineScheduler+DispatchQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue: CoroutineScheduler {
    
    @inlinable public func scheduleTask(_ task: @escaping () -> Void) {
        if self === DispatchQueue.main {
            Thread.isMainThread ? task() : async(execute: task)
        } else {
            async(execute: task)
        }
    }
    
}
