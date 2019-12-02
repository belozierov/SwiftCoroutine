//
//  DispatchQueue+Extensions.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension DispatchQueue {
    
    func delay(_ sec: Int) {
        _ = try? async { sleep(UInt32(sec)) }.await()
    }
    
}

func delay(_ sec: Int) {
    DispatchQueue.global().delay(sec)
}
