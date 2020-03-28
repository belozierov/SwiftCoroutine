//
//  TaskScheduler+DispatchQueue.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue: TaskScheduler {
    
    fileprivate enum Executor {
        case main((@escaping () -> Void) -> Void), notMain
    }
    
    public func executeTask(_ task: @escaping () -> Void) {
        switch getSpecific(key: .executorKey) {
        case .main(let scheduler): return scheduler(task)
        case .notMain: return async(execute: task)
        default: break
        }
        if self == .main {
            let scheduler = { (block: @escaping () -> Void) in
                pthread_main_np() != 0 ? block() : self.async(execute: block)
            }
            setSpecific(key: .executorKey, value: .main(scheduler))
            scheduler(task)
        } else {
            setSpecific(key: .executorKey, value: .notMain)
            async(execute: task)
        }
    }
    
}

extension DispatchSpecificKey where T == DispatchQueue.Executor {
    
    fileprivate static let executorKey = DispatchSpecificKey()
    
}